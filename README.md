# Rag LLM Gateway

> Async gRPC wrapper around the OpenAI Chat Completions API that streams tokens and rewrites follow-up questions into standalone retrieval queries.

---

## Responsibility

Implements `llm.v1.LLMGateway` with two RPCs: `AskLLM` (server-streaming chat completion that forwards each provider chunk as a `TextDelta`, terminated by exactly one `LLMDone`) and `RewriteQuery` (unary, history-aware query rewriter for retrieval). It centralises model selection, sampling parameters, and provider error translation (4xx -> INVALID_ARGUMENT, 429/5xx -> UNAVAILABLE). It does NOT cache responses, persist history, or run retrieval; callers pass the full prompt + optional `ChatMessage` history per call.

---

## Folder structure

```
Rag-llm-gateway/
├── server.py            # grpc.aio.server, reflection, async signal handlers
├── service.py           # LLMGatewayServicer: AskLLM + RewriteQuery
├── openai_client.py     # AsyncOpenAI streaming + non-streaming client wrapper
├── rewrite_prompt.py    # template for the standalone-query rewrite system prompt
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

| Path | Description |
|------|-------------|
| `server.py` | starts the async gRPC server on `GRPC_PORT` (default 50051) |
| `service.py` | proto <-> internal dataclass adapters, error mapping to grpc status codes |
| `openai_client.py` | yields `TextDelta` / `StreamDone` from OpenAI streaming chunks |
| `rewrite_prompt.py` | renders system + user prompt for the rewrite RPC |

---

## How to start

### Prerequisites
- Python 3.12
- Env vars: `OPENAI_API_KEY`, `GRPC_PORT`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_REWRITE_MODEL`, `LLM_REWRITE_TEMPERATURE`, `LLM_REWRITE_MAX_TOKENS`, `LOG_LEVEL`

### Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env

# 3. Run the service
python server.py
```

---

## Flow examples

1) Streaming a chat answer (`AskLLM`):
```
PromptRequest{
  system: "You answer using ONLY the provided context...",
  user:   "[1] ...chunk text...\n\nQuestion: What is the notice period?",
  history: [ {role: USER, content: "What is in the contract?"},
             {role: ASSISTANT, content: "It covers..."} ],
  model: "", temperature: 0, max_tokens: 0,         # 0 = use server defaults
  session_id: "s-9b2f"
}
-> stream LLMChunk{ text_delta: "Thirty " }
-> stream LLMChunk{ text_delta: "days." }
-> stream LLMChunk{ done: { input_tokens: 812, output_tokens: 14, finish_reason: "stop" } }
```

2) History-aware query rewrite (`RewriteQuery`):
```
RewriteQueryRequest{
  history: [ {USER, "What is in the contract?"},
             {ASSISTANT, "It covers termination, IP, and payment."} ],
  user_message: "And the notice period?"
}
-> RewriteQueryResponse{
     query: "contract termination notice period",
     input_tokens: 142, output_tokens: 7
   }
```

3) OpenAI error translation:
- HTTP 400/403 from OpenAI -> `grpc.StatusCode.INVALID_ARGUMENT`
- HTTP 429 or 5xx, or network error -> `grpc.StatusCode.UNAVAILABLE`
- empty rewrite output -> falls back to the raw `user_message`

---

## Service relationships

| Depends on | Why |
|------------|-----|
| OpenAI Chat Completions API | actual LLM inference (streaming + non-streaming) |

| Consumed by | Why |
|-------------|-----|
| Rag-orchestrator | calls `AskLLM` once per turn and `RewriteQuery` when prior history exists |

No event bus. Stateless: every request carries its own system, user, and history.
