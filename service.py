from __future__ import annotations

import logging

import grpc
from openai import APIError, APIStatusError

from llm.v1 import llm_pb2, llm_pb2_grpc
from openai_client import OpenAIStreamingClient, StreamDone, TextDelta

log = logging.getLogger("rag-llm-gateway")


class LLMGatewayServicer(llm_pb2_grpc.LLMGatewayServicer):
    def __init__(self, openai_client: OpenAIStreamingClient) -> None:
        self._openai = openai_client

    async def AskLLM(self, request, context):
        session_id = request.session_id
        system = request.system
        user = request.user

        log.info(
            "[llm-gateway] → AskLLM session_id=%r sys_len=%d user_len=%d model=%r",
            session_id,
            len(system),
            len(user),
            request.model or "<default>",
        )

        if not user:
            log.warning("[llm-gateway] AskLLM rejected: empty user prompt session_id=%r", session_id)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user prompt must not be empty")

        try:
            stream = self._openai.stream(
                system=system,
                user=user,
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
