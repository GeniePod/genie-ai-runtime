# HTTP Server

OpenAI-compatible REST API for the genie-ai-runtime inference engine.
Built on [cpp-httplib](https://github.com/yhirose/cpp-httplib) and
[nlohmann/json](https://github.com/nlohmann/json) (the same building
blocks llama.cpp's server uses), pulled at configure time via CMake
`FetchContent`. Single Engine instance, mutex-guarded `generate()` so
concurrent requests queue rather than corrupt KV state.

> **Opt-in build.** genie-ai-runtime ships as an embeddable engine.
> Build the standalone server with `-DJLLM_BUILD_SERVER=ON`:
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release -DJLLM_BUILD_SERVER=ON
> cmake --build build -j$(nproc)
> ```
>
> genie-claw consumes the engine via direct library link
> (`jetson_llm_core.a`); the server is for standalone REST deployments
> and A/B testing against `llama-server`.

Source: `src/main_server.cpp`, `src/server/http_server.cpp`.

## Run

```bash
./build/jetson-llm-server -m /path/to/model.gguf -p 8080
```

| Flag           | Default | Notes |
| -------------- | ------- | ----- |
| `-m PATH`      | —       | GGUF model file (required) |
| `-p PORT`      | 8080    | HTTP port |
| `-c INT`       | auto    | Context length (0 = auto from memory budget) |
| `--int8-kv`    | off     | Use INT8 KV cache. Server defaults to FP16 so Path F persistent KV save/load works — see [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67) for the v3 format that will unify the two. |
| `--fp16-kv`    | on      | FP16 KV (default for the server) |

## Endpoints

### `GET /health`

Jetson-specific snapshot (memory, thermal, power, GPU util).

```bash
curl -s http://jetson:8080/health
```

```json
{
  "status": "ok",
  "model": "Qwen3 4B Instruct Awq",
  "memory":  {"total_mb": 7619, "free_mb": 1730, "model_mb": 2685, "kv_mb": 576},
  "thermal": {"gpu_c": 58.5, "cpu_c": 56.0, "throttling": false},
  "power":   {"watts": 25, "gpu_mhz": 918, "gpu_max_mhz": 918},
  "gpu_util_pct": 72
}
```

### `GET /v1/models`

OpenAI-compatible model list (always one entry — the loaded model).

```bash
curl -s http://jetson:8080/v1/models
```

```json
{"object": "list",
 "data":   [{"id": "Qwen3 4B Instruct Awq", "object": "model", "owned_by": "genie-ai-runtime"}]}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat. Walks the full `messages[]` array through the
Qwen chat template; the rolling context determines the answer.

**Request body fields:**

| Field             | Type   | Default | Notes |
| ----------------- | ------ | ------- | ----- |
| `messages`        | array  | —       | Non-empty `[{role, content}]` (`role` ∈ `system` / `user` / `assistant`) |
| `max_tokens`      | int    | 256     | Generation cap; engine still stops on EOS |
| `temperature`     | float  | 0.7     | Sampling temperature |
| `top_k`           | int    | 40      | Top-k sampling |
| `top_p`           | float  | 0.9     | Top-p (nucleus) sampling |
| `stream`          | bool   | false   | Emit `text/event-stream` chunks instead of one JSON body |
| `think`           | bool   | false   | Allow Qwen3 thinking-mode output. When `true`, the model may emit `<think>reasoning</think>final answer` — the server splits the two and returns them in separate fields (see "Reasoning output" below). When `false`, an empty think block is prefixed to suppress reasoning. |
| `conversation_id` | string | —       | Path F persistent KV id (`[A-Za-z0-9_-]{1,64}`); only effective under FP16 KV today |
| `kv_int8`         | bool   | (load)  | Override KV format per-request. Defaults to whatever the server was loaded with; mismatch with load-time format produces garbage tokens. Don't surface to clients today — see [#67](https://github.com/GeniePod/genie-ai-runtime/issues/67). |

**Non-streaming response** (typical):

```bash
curl -s http://jetson:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],
       "max_tokens":40}'
```

```json
{
  "id": "jllm-1778916536",
  "object": "chat.completion",
  "created": 1778916536,
  "model": "Qwen3 4B Instruct Awq",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "2 + 2 equals 4."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 17, "completion_tokens": 7, "total_tokens": 24},
  "jetson": {
    "decode_tok_s": 11.6, "prompt_tok_s": 37.6, "ttft_ms": 914,
    "peak_mem_mb": 0, "peak_temp_c": 0.0
  }
}
```

The `jetson` block is a non-standard extension with Jetson-specific
perf metrics. (`peak_mem_mb` / `peak_temp_c` currently report 0 on the
server path — the decode-side live-stats watcher used by the CLI isn't
attached yet. Cosmetic; tracked separately.)

**Streaming** — set `"stream": true` and read `text/event-stream`:

```bash
curl -N -s http://jetson:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"count to five"}],
       "max_tokens":40, "stream":true}'
```

```
data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],
       "id":"jllm-...","object":"chat.completion.chunk","model":"..."}

data: {"choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}], ...}
data: {"choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}], ...}
data: {"choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null}], ...}
...
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}], ...}

data: [DONE]
```

OpenAI-shape `chat.completion.chunk` envelopes — clients written
against OpenAI's stream protocol work as-is.

### Reasoning output (`think: true`)

Qwen3 supports a thinking/reasoning mode. With `"think": true` in the
request, the model may produce `<think>chain-of-thought</think>final
answer`. The server splits the two and surfaces them in separate
fields. Shape mirrors DeepSeek's reasoning API (`reasoning_content`),
which most OpenAI-compatible clients tolerate.

**Non-streaming:**

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The answer is 4.",
      "reasoning_content": "Let me add 2 and 2. 2 + 2 = 4. So the answer is 4."
    },
    "finish_reason": "stop"
  }],
  ...
}
```

`reasoning_content` is omitted when empty (e.g. `think: false` requests,
or `think: true` requests where the model chose to skip reasoning).

**Streaming:** reasoning chunks and content chunks arrive as separate
deltas, in order:

```
data: {"choices":[{"delta":{"role":"assistant"}, ...}]}
data: {"choices":[{"delta":{"reasoning_content":"Let me"}, ...}]}
data: {"choices":[{"delta":{"reasoning_content":" add"}, ...}]}
...
data: {"choices":[{"delta":{"content":"The"}, ...}]}
data: {"choices":[{"delta":{"content":" answer"}, ...}]}
...
data: {"choices":[{"delta":{},"finish_reason":"stop"}, ...]}
data: [DONE]
```

Currently Qwen3-specific (looks for literal `<think>` / `</think>`
tokens). Other reasoning-model templates can dispatch on
`engine.config().name` when added.

### Errors

```json
{"error": {"message": "messages: non-empty array required",
           "type": "invalid_request_error", "code": 400}}
```

- `400 invalid_request_error` — malformed JSON or empty `messages`
- `501 not_implemented` — reserved (no current 501 paths)

## Client examples

### Python — `requests` (non-streaming)

```python
import requests

r = requests.post("http://jetson:8080/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
})
print(r.json()["choices"][0]["message"]["content"])
```

### Python — `openai` SDK (streaming)

```python
from openai import OpenAI
c = OpenAI(base_url="http://jetson:8080/v1", api_key="unused")
for ev in c.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "count to 5"}],
    max_tokens=40, stream=True,
):
    print(ev.choices[0].delta.content or "", end="", flush=True)
print()
```

### curl — `/health` monitoring

```bash
watch -n5 'curl -s http://jetson:8080/health | python3 -m json.tool'
```

## Running as a systemd service

Quick install after a successful `-DJLLM_BUILD_SERVER=ON` build:

```bash
sudo ./scripts/setup.sh
# optional flags: --user pat --model /path.gguf --port 8080
sudo systemctl enable --now jetson-llm-server
sudo journalctl -u jetson-llm-server -f
```

What it installs:

| Path                                             | Notes |
| ------------------------------------------------ | ----- |
| `/opt/jetson-llm/bin/jetson-llm-server`          | The binary |
| `/etc/systemd/system/jetson-llm-server.service`  | Unit (templated `User=`) |
| `/etc/jetson-llm/server.env`                     | `MODEL_PATH`, `PORT` — edit + `systemctl restart` |

Unit highlights:

- `Restart=on-failure` with a 3-restart / 120 s burst cap (no flap when the model file is missing).
- `TimeoutStartSec=120` — cold load is NVMe-bound (~30 s for a 2.4 GB GGUF on a fresh boot pagecache).
- `LimitMEMLOCK=infinity` — CUDA pinned memory + large model mmap.
- `KillSignal=SIGTERM` — the engine catches it and stops generation cleanly.
- **No `ProtectSystem=strict`** — `/dev/nvgpu*` + Tegra ioctls would be blocked. `NoNewPrivileges=true` and `PrivateTmp=true` are on.
- `User=` is rewritten by `setup.sh` to whoever ran the installer (`SUDO_USER` if available, else `id -un`).

## Implementation notes

- **One Engine, mutex-guarded `generate()`.** cpp-httplib runs handlers
  on its worker-thread pool, so racing requests serialize at the
  engine boundary — concurrent inference would corrupt the KV pool.
  The mutex is taken inside the SSE chunked-content-provider lambda so
  response headers + first chunk hit the wire before we block.

- **Client-disconnect cancels generation.** `sink.write` returns false
  on EOF; the streaming `token_cb` calls `engine.stop()` so the decode
  loop exits at its next checkpoint instead of running to `max_tokens`
  for a request nobody's reading.

- **EOS marker filtered out.** The engine fires
  `token_cb(text="<|im_end|>", is_eos=true)` for the EOS special
  token; the callback drops it so user-visible content stops at the
  last real token. The `finish_reason: "stop"` chunk conveys EOS.

- **Timeouts** bumped from cpp-httplib's 5 s/5 s defaults to 30 s read
  / 600 s write — long completions need it. LAN-only deployment so
  slow-loris isn't in scope.

## Roadmap

- **#67** — Path F format v3, so INT8 KV + persistent KV interop.
- **#5** acceptance items remaining: parallel-A/B vs `llama-server` once
  systemd unit is verified under load (not on the genie-claw critical
  path since the embedding API is the consumer that matters).
- Streaming `usage` block (OpenAI added it to their stream protocol
  recently) — easy add when a consumer needs it.
- `peak_mem_mb` / `peak_temp_c` on the server path — attach the
  decode-side live-stats watcher the CLI uses.
