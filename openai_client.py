from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator

from openai import AsyncOpenAI

log = logging.getLogger("rag-llm-gateway.openai_client")


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class StreamDone:
    input_tokens: int
    output_tokens: int
    finish_reason: str


StreamEvent = TextDelta | StreamDone


class OpenAIStreamingClient:
    """Thin wrapper around AsyncOpenAI chat.completions streaming.

    Yields TextDelta for each non-empty `delta.content` and a single
    terminal StreamDone carrying usage + finish reason.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=api_key)
        self._default_model = default_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._default_temperature = (
            default_temperature
            if default_temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.2"))
        )
        self._default_max_tokens = (
            default_max_tokens
            if default_max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", "1024"))
        )
        log.info(
            "OpenAIStreamingClient ready model=%r temperature=%.2f max_tokens=%d",
            self._default_model,
            self._default_temperature,
            self._default_max_tokens,
        )

    async def stream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        chosen_model = model or self._default_model
        chosen_temperature = (
            temperature if temperature and temperature > 0 else self._default_temperature
        )
        chosen_max_tokens = (
            max_tokens if max_tokens and max_tokens > 0 else self._default_max_tokens
        )

        log.info(
            "[openai] → chat.completions stream session_id=%r model=%r temp=%.2f max_tokens=%d sys_len=%d user_len=%d",
            session_id,
            chosen_model,
            chosen_temperature,
            chosen_max_tokens,
            len(system),
            len(user),
        )

        stream = await self._client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        finish_reason = "stop"
        input_tokens = 0
        output_tokens = 0
        delta_count = 0

        async for chunk in stream:
            # The final chunk with usage has empty choices but non-null usage.
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            text = (choice.delta.content or "") if choice.delta else ""
            if text:
                delta_count += 1
                log.debug(
                    "[openai] ← delta[%d] session_id=%r text=%r",
                    delta_count,
                    session_id,
                    text,
                )
                yield TextDelta(text=text)

        log.info(
            "[openai] ← stream complete session_id=%r deltas=%d input_tokens=%d output_tokens=%d finish_reason=%r",
            session_id,
            delta_count,
            input_tokens,
            output_tokens,
            finish_reason,
        )
        yield StreamDone(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        await self._client.close()
