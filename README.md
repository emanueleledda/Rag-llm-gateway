# Rag-llm-gateway

gRPC microservice that owns LLM provider communication for the Rag platform.
Exposes a single server-streaming RPC, `llm.v1.LLMGateway/AskLLM`, that
forwards prompt chunks from OpenAI's chat completions streaming API.

## Contract

See [`Rag-proto-contract/proto/llm/v1/llm.proto`](../Rag-proto-contract/proto/llm/v1/llm.proto).

`AskLLM(PromptRequest) returns (stream LLMChunk)`:

- `PromptRequest`: `system`, `user`, optional `model`, `temperature`,
  `max_tokens`, `session_id` (correlation id).
- `LLMChunk`: a oneof with either a `text_delta` (incremental token) or a
  terminal `done` (usage + finish reason). The stream terminates with
  exactly one `done` chunk.

## Configuration

| Env var            | Default        | Notes                                |
|--------------------|----------------|--------------------------------------|
| `GRPC_PORT`        | `50051`        | Server bind port.                    |
| `OPENAI_API_KEY`   | _required_     | OpenAI API key.                      |
| `LLM_MODEL`        | `gpt-4o-mini`  | Default model when request omits it. |
| `LLM_TEMPERATURE`  | `0.2`          | Default sampling temperature.        |
| `LLM_MAX_TOKENS`   | `1024`         | Default max output tokens.           |
| `LOG_LEVEL`        | `INFO`         | `DEBUG` for verbose chunk logging.   |

## Local run

```bash
pip install -r requirements.txt
python -m grpc_tools.protoc \
  -I ../Rag-proto-contract/proto \
  --python_out=. --grpc_python_out=. --pyi_out=. \
  llm/v1/llm.proto
OPENAI_API_KEY=sk-... python server.py
```
