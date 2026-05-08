# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- deps ----
FROM base AS deps
COPY Rag-llm-gateway/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements.txt

# ---- proto stubs ----
FROM deps AS proto
COPY Rag-proto-contract/proto/ /app/proto-src/
RUN mkdir -p /app/generated && \
    python -m grpc_tools.protoc \
      -I /app/proto-src \
      --python_out=/app/generated \
      --grpc_python_out=/app/generated \
      --pyi_out=/app/generated \
      llm/v1/llm.proto && \
    touch /app/generated/__init__.py \
          /app/generated/llm/__init__.py \
          /app/generated/llm/v1/__init__.py

# ---- runtime ----
FROM proto AS runtime
WORKDIR /app
COPY Rag-llm-gateway/server.py /app/server.py
COPY Rag-llm-gateway/service.py /app/service.py
COPY Rag-llm-gateway/openai_client.py /app/openai_client.py

ENV PYTHONPATH=/app:/app/generated \
    GRPC_PORT=50051

EXPOSE 50051

CMD ["python", "server.py"]
