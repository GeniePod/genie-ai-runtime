// http_server.cpp — OpenAI-compatible REST API for Jetson LLM
//
// Minimal HTTP server using raw sockets (no external dependency).
// Supports:
//   POST /v1/chat/completions  — OpenAI-compatible chat
//   GET  /health               — Jetson-specific health metrics
//   GET  /v1/models            — List loaded model
//
// For production: replace with cpp-httplib or Crow for proper HTTP parsing.
// This implementation handles the happy path for quick deployment.

#include "jllm_engine.h"
#include "jllm_jetson.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

namespace jllm {

// ── Simple JSON helpers (no dependency) ──────────────────────────────────

static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + search.size() + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static std::string extract_last_content(const std::string& json) {
    // Find last "content":"..." in the messages array
    std::string key = "\"content\"";
    size_t pos = json.rfind(key);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + key.size() + 1);
    if (pos == std::string::npos) return "";
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

// ── HTTP response helpers ────────────────────────────────────────────────

static void send_response(int client, int code, const std::string& content_type,
                          const std::string& body) {
    char header[512];
    snprintf(header, sizeof(header),
             "HTTP/1.1 %d OK\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n\r\n",
             code, content_type.c_str(), body.size());
    write(client, header, strlen(header));
    write(client, body.data(), body.size());
}

// ── Health endpoint ──────────────────────────────────────────────────────

static std::string build_health_json(Engine& engine) {
    auto ls = engine.stats();
    auto ts = read_thermal();
    auto ps = read_power_state();
    auto budget = engine.memory();

    char buf[1024];
    snprintf(buf, sizeof(buf),
        "{"
        "\"status\":\"ok\","
        "\"model\":\"%s\","
        "\"memory\":{\"total_mb\":%ld,\"free_mb\":%ld,\"model_mb\":%ld,\"kv_mb\":%ld},"
        "\"thermal\":{\"gpu_c\":%.1f,\"cpu_c\":%.1f,\"throttling\":%s},"
        "\"power\":{\"mode\":\"%dW\",\"gpu_mhz\":%d,\"gpu_max_mhz\":%d},"
        "\"gpu_util_pct\":%d"
        "}",
        engine.config().name.c_str(),
        budget.total_mb, budget.free_mb(), budget.model_mb, budget.kv_cache_mb,
        ts.gpu_temp_c, ts.cpu_temp_c, ts.throttling ? "true" : "false",
        ps.watts, ps.gpu_freq_mhz, ps.gpu_freq_max_mhz,
        ls.gpu_util_pct);
    return buf;
}

// ── Chat completion endpoint ─────────────────────────────────────────────

static std::string build_completion_json(Engine& engine, const std::string& prompt,
                                          const GenParams& params) {
    std::string response;
    auto stats = engine.generate(prompt, params, [&](const char* text, bool eos) {
        response += text;
    });

    char buf[4096];
    snprintf(buf, sizeof(buf),
        "{"
        "\"id\":\"jllm-%ld\","
        "\"object\":\"chat.completion\","
        "\"model\":\"%s\","
        "\"choices\":[{"
        "\"index\":0,"
        "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
        "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{"
        "\"prompt_tokens\":%d,"
        "\"completion_tokens\":%d,"
        "\"total_tokens\":%d"
        "},"
        "\"jetson\":{\"decode_tok_s\":%.1f,\"peak_mem_mb\":%ld,\"peak_temp_c\":%.1f}"
        "}",
        time(nullptr),
        engine.config().name.c_str(),
        json_escape(response).c_str(),
        stats.prompt_tokens, stats.completion_tokens,
        stats.prompt_tokens + stats.completion_tokens,
        stats.decode_tok_per_sec, stats.peak_memory_mb, stats.peak_thermal_c);
    return buf;
}

// ── Request handler ──────────────────────────────────────────────────────

static void handle_client(int client, Engine& engine) {
    char buf[8192] = {};
    int n = read(client, buf, sizeof(buf) - 1);
    if (n <= 0) { close(client); return; }

    std::string request(buf, n);

    // Parse method and path
    std::string method, path;
    if (request.substr(0, 3) == "GET") {
        method = "GET";
        size_t sp = request.find(' ', 4);
        path = request.substr(4, sp - 4);
    } else if (request.substr(0, 4) == "POST") {
        method = "POST";
        size_t sp = request.find(' ', 5);
        path = request.substr(5, sp - 5);
    }

    // Route
    if (method == "GET" && path == "/health") {
        send_response(client, 200, "application/json", build_health_json(engine));
    }
    else if (method == "GET" && path == "/v1/models") {
        char models[256];
        snprintf(models, sizeof(models),
                 "{\"data\":[{\"id\":\"%s\",\"object\":\"model\"}]}",
                 engine.config().name.c_str());
        send_response(client, 200, "application/json", models);
    }
    else if (method == "POST" && path == "/v1/chat/completions") {
        // Extract body (after \r\n\r\n)
        size_t body_start = request.find("\r\n\r\n");
        std::string body = (body_start != std::string::npos) ? request.substr(body_start + 4) : "";

        std::string prompt = extract_last_content(body);
        if (prompt.empty()) prompt = "Hello";

        GenParams params;
        // TODO: parse max_tokens, temperature from JSON body

        // Path F (#45): pick up an optional conversation_id from the
        // request body. Validate before forwarding; an invalid id is
        // dropped to single-shot mode rather than rejected as a 4xx so
        // existing OpenAI-compatible clients don't break if they pass
        // something unexpected.
        std::string conv_id = extract_json_string(body, "conversation_id");
        if (!conv_id.empty()) {
            if (validate_conversation_id(conv_id)) {
                params.conversation_id = conv_id;
            } else {
                fprintf(stderr,
                        "[http] WARNING: conversation_id '%s' rejected — "
                        "must match [A-Za-z0-9_-]{1,64}\n", conv_id.c_str());
            }
        }

        send_response(client, 200, "application/json",
                     build_completion_json(engine, prompt, params));
    }
    else {
        send_response(client, 404, "application/json", "{\"error\":\"not found\"}");
    }

    close(client);
}

// ── Server main loop ─────────────────────────────────────────────────────

void run_server(Engine& engine, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return; }
    if (listen(server_fd, 8) < 0) { perror("listen"); return; }

    fprintf(stderr, "\n╔══════════════════════════════════════════╗\n");
    fprintf(stderr, "║  jetson-llm server on port %d           ║\n", port);
    fprintf(stderr, "║  Health:  GET  http://0.0.0.0:%d/health ║\n", port);
    fprintf(stderr, "║  Chat:    POST http://0.0.0.0:%d/v1/... ║\n", port);
    fprintf(stderr, "╚══════════════════════════════════════════╝\n\n");

    while (true) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client = accept(server_fd, (sockaddr*)&client_addr, &client_len);
        if (client < 0) continue;

        // Handle each request (sequential — one inference at a time on Jetson)
        handle_client(client, engine);
    }
}

}  // namespace jllm
