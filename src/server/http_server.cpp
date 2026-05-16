// http_server.cpp — OpenAI-compatible REST API for Jetson LLM
//
// Built on cpp-httplib (HTTP) + nlohmann/json (JSON) — same stack
// llama.cpp's server uses. Single Engine instance, mutex-guarded
// generate() so racing requests queue rather than corrupting KV state.
//
// Endpoints:
//   GET  /health                  Jetson system health snapshot
//   GET  /v1/models               List loaded model (OpenAI shape)
//   POST /v1/chat/completions     OpenAI-compatible chat (non-streaming)
//
// Streaming (`stream:true`) returns 501 today — wired up in issue #5
// phase 3 (separate PR).

#include "jllm_engine.h"
#include "jllm_jetson.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

using json = nlohmann::json;

namespace jllm {

// One GPU, one engine. cpp-httplib spawns a thread per request, so we
// serialize calls into generate() — overlapping inference would just
// stomp on the KV cache pool.
static std::mutex g_engine_mutex;

// ── /health response ─────────────────────────────────────────────────────

static json build_health(const Engine& engine) {
    auto ls = engine.stats();
    auto ts = read_thermal();
    auto ps = read_power_state();
    auto budget = engine.memory();
    return json{
        {"status", "ok"},
        {"model", engine.config().name},
        {"memory", {
            {"total_mb", budget.total_mb},
            {"free_mb",  budget.free_mb()},
            {"model_mb", budget.model_mb},
            {"kv_mb",    budget.kv_cache_mb},
        }},
        {"thermal", {
            {"gpu_c",      ts.gpu_temp_c},
            {"cpu_c",      ts.cpu_temp_c},
            {"throttling", ts.throttling},
        }},
        {"power", {
            {"watts",       ps.watts},
            {"gpu_mhz",     ps.gpu_freq_mhz},
            {"gpu_max_mhz", ps.gpu_freq_max_mhz},
        }},
        {"gpu_util_pct", ls.gpu_util_pct},
    };
}

// ── Qwen chat template ───────────────────────────────────────────────────
// Currently the only template supported — GeniePod runs Qwen3 today.
// When we add Phi / Gemma / Llama-3 we'll dispatch on engine.config().name.

static std::string format_qwen_chat(const json& messages, bool think) {
    std::string out;
    for (const auto& m : messages) {
        const std::string role    = m.value("role",    "user");
        const std::string content = m.value("content", "");
        out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n";
    if (!think) out += "<think>\n\n</think>\n\n";
    return out;
}

// ── Chat completion (non-streaming) ──────────────────────────────────────

static json build_completion(Engine& engine, const std::string& prompt,
                              const GenParams& params) {
    std::string response;
    auto stats = engine.generate(prompt, params, [&](const char* text, bool eos) {
        // Skip the EOS marker's decoded text (`<|im_end|>` for Qwen) so it
        // doesn't end up tail-glued to the user-visible content.
        if (eos) return;
        response += text;
    });

    return json{
        {"id",      "jllm-" + std::to_string(std::time(nullptr))},
        {"object",  "chat.completion"},
        {"created", (int64_t)std::time(nullptr)},
        {"model",   engine.config().name},
        {"choices", json::array({
            json{
                {"index", 0},
                {"message", {{"role", "assistant"}, {"content", response}}},
                {"finish_reason", "stop"},
            }
        })},
        {"usage", {
            {"prompt_tokens",     stats.prompt_tokens},
            {"completion_tokens", stats.completion_tokens},
            {"total_tokens",      stats.prompt_tokens + stats.completion_tokens},
        }},
        {"jetson", {
            {"decode_tok_s", stats.decode_tok_per_sec},
            {"prompt_tok_s", stats.prompt_tok_per_sec},
            {"ttft_ms",      stats.ttft_ms},
            {"peak_mem_mb",  stats.peak_memory_mb},
            {"peak_temp_c",  stats.peak_thermal_c},
        }},
    };
}

// Reply with `{ "error": { "message": ..., "type": ... } }` (OpenAI shape).
static void send_error(httplib::Response& res, int code,
                       const std::string& msg, const std::string& type) {
    res.status = code;
    res.set_content(json{{"error",
        json{{"message", msg}, {"type", type}, {"code", code}}}}.dump(),
        "application/json");
}

// ── SSE streaming for chat completions ───────────────────────────────────

// One "data: {chat.completion.chunk}\n\n" envelope.
static std::string sse_chunk(const std::string& id, int64_t created,
                              const std::string& model,
                              const json& delta, const char* finish_reason) {
    return "data: " + json{
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", created},
        {"model",   model},
        {"choices", json::array({json{
            {"index", 0},
            {"delta", delta},
            {"finish_reason", finish_reason ? json(finish_reason) : json(nullptr)},
        }})},
    }.dump() + "\n\n";
}

// Streaming /v1/chat/completions. Holds the engine mutex for the whole
// stream so concurrent SSE requests serialize on the single GPU.
// cpp-httplib calls the content provider on its worker thread; we run
// engine.generate() synchronously inside and push each token through
// DataSink as soon as token_cb fires.
//
// `prompt`, `params`, `model_name` are captured by value so the lambda's
// lifetime is independent of the outer handler scope (cpp-httplib may
// dispatch the provider on a different thread / after the handler
// returns, depending on version).
static void stream_chat_completion(httplib::Response& res, Engine& engine,
                                    const std::string& prompt,
                                    const GenParams& params,
                                    const std::string& model_name) {
    res.set_header("Cache-Control",      "no-cache");
    res.set_header("Connection",         "keep-alive");
    res.set_header("X-Accel-Buffering",  "no");   // disable nginx buffering

    res.set_chunked_content_provider("text/event-stream",
        [&engine, prompt, params, model_name]
        (size_t /*offset*/, httplib::DataSink& sink) -> bool {
            std::lock_guard<std::mutex> lk(g_engine_mutex);

            const std::string id      = "jllm-" + std::to_string(std::time(nullptr));
            const int64_t     created = std::time(nullptr);

            // Role-only opener. Mirrors OpenAI's own first chunk —
            // python `openai` client needs the role delta first.
            std::string open = sse_chunk(id, created, model_name,
                                          json{{"role", "assistant"}}, nullptr);
            if (!sink.write(open.c_str(), open.size())) {
                sink.done();
                return false;
            }

            // Per-token deltas. If client disconnects, sink.write returns
            // false — we set client_alive and call engine.stop() so the
            // decode loop exits at its next checkpoint instead of running
            // to max_tokens for a request nobody's reading.
            bool client_alive = true;
            auto token_cb = [&](const char* text, bool eos) {
                if (!client_alive) return;
                // Don't stream the EOS marker's decoded text — the
                // separate finish_reason:"stop" chunk below conveys it.
                if (eos) return;
                std::string chunk = sse_chunk(id, created, model_name,
                                               json{{"content", text}}, nullptr);
                if (!sink.write(chunk.c_str(), chunk.size())) {
                    client_alive = false;
                    engine.stop();
                }
            };

            bool errored = false;
            try {
                (void)engine.generate(prompt, params, token_cb);
            } catch (const std::exception& e) {
                errored = true;
                if (client_alive) {
                    std::string err = "data: " + json{{"error", json{
                        {"message", e.what()},
                        {"type",    "internal_error"}
                    }}}.dump() + "\n\n";
                    sink.write(err.c_str(), err.size());
                }
            }

            if (client_alive) {
                if (!errored) {
                    std::string fin = sse_chunk(id, created, model_name,
                                                 json::object(), "stop");
                    sink.write(fin.c_str(), fin.size());
                }
                std::string done = "data: [DONE]\n\n";
                sink.write(done.c_str(), done.size());
            }
            sink.done();
            return false;   // single-pass: provider called once
        });
}

// ── Server entry ─────────────────────────────────────────────────────────

// `default_kv_int8` is the KV cache format the Engine was loaded with —
// must match what generate() reads from the pool, or the attention path
// reinterprets the KV bytes against the wrong scale layout and the
// model emits garbage. Set from main_server.cpp's --int8-kv / --fp16-kv.
// Per-request `"kv_int8": <bool>` can still override (intended for the
// future case where we want to surface it for diagnostics; today both
// values produce identical results since the pool format is fixed at
// load-time — overriding here just means "engine sees the matching
// flag" — but if it ever drifts from the load-time value the attention
// path will produce wrong output, so don't surface this on the client
// side until #67 lands and the format truly becomes selectable per
// request).
void run_server(Engine& engine, int port, bool default_kv_int8) {
    httplib::Server svr;
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}});

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(build_health(engine).dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{
            {"object", "list"},
            {"data",   json::array({json{
                {"id",       engine.config().name},
                {"object",   "model"},
                {"owned_by", "genie-ai-runtime"},
            }})},
        }.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions",
             [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("invalid json: ") + e.what(),
                       "invalid_request_error");
            return;
        }

        json messages = body.value("messages", json::array());
        if (!messages.is_array() || messages.empty()) {
            send_error(res, 400, "messages: non-empty array required",
                       "invalid_request_error");
            return;
        }

        GenParams params;
        params.max_tokens  = body.value("max_tokens",  256);
        params.temperature = body.value("temperature", 0.7f);
        params.top_k       = body.value("top_k",       40);
        params.top_p       = body.value("top_p",       0.9f);
        // CRITICAL: default to the load-time value, not the GenParams
        // struct default — alpha.12 flipped the struct default to true,
        // but the server keeps FP16 KV by default for Path F interop.
        // Mismatch here = the decode path reinterprets KV bytes against
        // the wrong format and the model emits garbage tokens.
        params.kv_int8 = body.value("kv_int8", default_kv_int8);

        std::string conv_id = body.value("conversation_id", "");
        if (!conv_id.empty()) {
            if (validate_conversation_id(conv_id)) {
                params.conversation_id = conv_id;
            } else {
                fprintf(stderr,
                        "[http] WARNING: conversation_id '%s' rejected — "
                        "must match [A-Za-z0-9_-]{1,64}\n", conv_id.c_str());
            }
        }

        const bool think = body.value("think", false);
        const std::string prompt = format_qwen_chat(messages, think);

        if (body.value("stream", false)) {
            // SSE path — mutex is taken inside the chunked-content
            // provider so the response headers + first chunk go out
            // immediately, before we block on engine.generate().
            stream_chat_completion(res, engine, prompt, params,
                                    engine.config().name);
            return;
        }

        std::lock_guard<std::mutex> lk(g_engine_mutex);
        res.set_content(build_completion(engine, prompt, params).dump(),
                        "application/json");
    });

    // OPTIONS for CORS preflight — needed for browser-based callers.
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    // cpp-httplib defaults are 5 s read / 5 s write. Way too short — a
    // 256-token completion at 10 tok/s is already 25 s. Bump both
    // generously; LAN-only deployment so we don't worry about slow
    // attacker hold-open.
    svr.set_read_timeout(  30, 0);   // request body (small JSON)
    svr.set_write_timeout(600, 0);   // response stream (long completions)
    svr.set_idle_interval(  0, 100); // 100 ms keep-alive idle poll

    fprintf(stderr,
        "\n[server] listening on http://0.0.0.0:%d\n"
        "[server]   GET  /health\n"
        "[server]   GET  /v1/models\n"
        "[server]   POST /v1/chat/completions   (stream:true → SSE)\n\n",
        port);

    if (!svr.listen("0.0.0.0", port)) {
        fprintf(stderr, "[server] FATAL: failed to bind 0.0.0.0:%d\n", port);
    }
}

}  // namespace jllm
