from __future__ import annotations

import logging

import grpc
from openai import APIError, APIStatusError

from llm.v1 import llm_pb2, llm_pb2_grpc
from openai_client import (
    ChatTurn,
    OpenAIStreamingClient,
    StreamDone,
    TextDelta,
)
from rewrite_prompt import render_rewrite_prompt

log = logging.getLogger("rag-llm-gateway")


# Map proto MessageRole → OpenAI role string. Anything not in this map is
# treated as "user" by the OpenAI client (see _build_messages).
_PROTO_TO_OPENAI_ROLE = {
    llm_pb2.MESSAGE_ROLE_USER: "user",
    llm_pb2.MESSAGE_ROLE_ASSISTANT: "assistant",
}


def _proto_history_to_turns(history) -> list[ChatTurn]:
    """Translate proto ChatMessage list → internal ChatTurn list.

    Empty content is dropped here as well (defense in depth — the OpenAI
    client also drops empties).
    """
    turns: list[ChatTurn] = []
    for msg in history:
        role = _PROTO_TO_OPENAI_ROLE.get(msg.role, "user")
        content = msg.content or ""
        if not content.strip():
            continue
        turns.append(ChatTurn(role=role, content=content))
    return turns


class LLMGatewayServicer(llm_pb2_grpc.LLMGatewayServicer):
    def __init__(self, openai_client: OpenAIStreamingClient) -> None:
        self._openai = openai_client

    async def AskLLM(self, request, context):
        session_id = request.session_id
        system = request.system
        user = request.user
        history = _proto_history_to_turns(request.history)

        log.info(
            "[llm-gateway] → AskLLM session_id=%r sys_len=%d user_len=%d history_len=%d model=%r",
            session_id,
            len(system),
            len(user),
            len(history),
            request.model or "<default>",
        )

        if not user:
            log.warning("[llm-gateway] AskLLM rejected: empty user prompt session_id=%r", session_id)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user prompt must not be empty")

        try:
            stream = self._openai.stream(
                system=system,
                user=user,
                history=history,
                model=request.model or None,
                temperature=request.temperature or None,
                max_tokens=request.max_tokens or None,
                session_id=session_id,
            )
            async for event in stream:
                if isinstance(event, TextDelta):
                    yield llm_pb2.LLMChunk(
                        text_delta=llm_pb2.TextDelta(text=event.text),
                    )
                elif isinstance(event, StreamDone):
                    yield llm_pb2.LLMChunk(
                        done=llm_pb2.LLMDone(
                            input_tokens=event.input_tokens,
                            output_tokens=event.output_tokens,
                            finish_reason=event.finish_reason,
                        ),
                    )
        except APIStatusError as exc:
            log.warning(
                "[llm-gateway] OpenAI APIStatusError session_id=%r status=%s body=%s",
                session_id,
                exc.status_code,
                exc.message,
            )
            code = (
                grpc.StatusCode.INVALID_ARGUMENT
                if 400 <= exc.status_code < 500 and exc.status_code != 429
                else grpc.StatusCode.UNAVAILABLE
            )
            await context.abort(code, f"openai: {exc.message}")
        except APIError as exc:
            log.warning(
                "[llm-gateway] OpenAI APIError session_id=%r message=%s",
                session_id,
                exc.message,
            )
            await context.abort(grpc.StatusCode.UNAVAILABLE, f"openai: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            log.exception("[llm-gateway] AskLLM unexpected failure session_id=%r", session_id)
            await context.abort(grpc.StatusCode.INTERNAL, f"llm-gateway: {exc}")

        log.info("[llm-gateway] ← AskLLM stream complete session_id=%r", session_id)

    async def RewriteQuery(self, request, context):
        """Rewrite the latest user message into a standalone retrieval query.

        Caller passes recent conversation history + the new user message.
        We render the rewrite prompt template, run a non-streaming
        completion against the rewrite model, and return the trimmed
        result. Empty rewrites fall back to the original message — better
        to retrieve on the raw text than on nothing.
        """
        session_id = request.session_id
        user_message = request.user_message or ""

        if not user_message.strip():
            log.warning("[llm-gateway] RewriteQuery rejected: empty user_message session_id=%r", session_id)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_message must not be empty")

        log.info(
            "[llm-gateway] → RewriteQuery session_id=%r history_len=%d user_message_len=%d",
            session_id,
            len(request.history),
            len(user_message),
        )

        rendered = render_rewrite_prompt(request.history, user_message)

        try:
            result = await self._openai.complete(
                system=rendered.system,
                user=rendered.user,
                model=request.model or None,
                temperature=request.temperature or None,
                max_tokens=request.max_tokens or None,
                session_id=session_id,
            )
        except APIStatusError as exc:
            log.warning(
                "[llm-gateway] RewriteQuery OpenAI APIStatusError session_id=%r status=%s body=%s",
                session_id,
                exc.status_code,
                exc.message,
            )
            code = (
                grpc.StatusCode.INVALID_ARGUMENT
                if 400 <= exc.status_code < 500 and exc.status_code != 429
                else grpc.StatusCode.UNAVAILABLE
            )
            await context.abort(code, f"openai: {exc.message}")
        except APIError as exc:
            log.warning(
                "[llm-gateway] RewriteQuery OpenAI APIError session_id=%r message=%s",
                session_id,
                exc.message,
            )
            await context.abort(grpc.StatusCode.UNAVAILABLE, f"openai: {exc.message}")
        except Exception as exc:  # noqa: BLE001
            log.exception("[llm-gateway] RewriteQuery unexpected failure session_id=%r", session_id)
            await context.abort(grpc.StatusCode.INTERNAL, f"llm-gateway: {exc}")

        # Strip surrounding quotes/whitespace some models add despite instructions.
        rewritten = result.text.strip().strip('"').strip("'").strip()
        if not rewritten:
            # Fallback: rather than returning an empty query (retrieval would
            # likely 400 or return zero hits), surface the raw user message.
            log.warning(
                "[llm-gateway] RewriteQuery produced empty output session_id=%r — falling back to raw user_message",
                session_id,
            )
            rewritten = user_message.strip()

        log.info(
            "[llm-gateway] ← RewriteQuery session_id=%r out_len=%d input_tokens=%d output_tokens=%d",
            session_id,
            len(rewritten),
            result.input_tokens,
            result.output_tokens,
        )
        return llm_pb2.RewriteQueryResponse(
            query=rewritten,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
