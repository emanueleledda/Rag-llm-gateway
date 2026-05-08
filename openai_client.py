from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

log = logging.getLogger("rag-llm-gateway.openai_client")


# ---- Internal value objects -------------------------------------------------
#
# These dataclasses are how the gRPC service layer talks to the OpenAI client.
# Keeping them gateway-local (not proto types) means this module never imports
# generated stubs and stays trivially testable.


@dataclass(frozen=True)
class ChatTurn:
    """One historical message passed alongside the current user prompt.

    `role` must be "user" or "assistant" — anything else is coerced to "user"
    by `_build_messages` since OpenAI rejects unknown roles.
    """
    role: str
    content: str


@dataclass(frozen=True)
class TextDelta:
    """A single token (or short text fragment) streamed from the model."""
    text: str


@dataclass(frozen=True)
class StreamDone:
    """Terminal event of a streaming completion. Carries usage + finish reason."""
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class CompletionResult:
    """Return shape for the non-streaming `complete()` call (used by RewriteQuery)."""
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


StreamEvent = TextDelta | StreamDone


def _build_messages(
    *,
    system: str,
    history: Sequence[ChatTurn] | None,
    user: str,
) -> list[dict]:
    """Assemble the OpenAI `messages=[...]` list.

    Order is: system → history (chronological) → current user message.
    OpenAI treats the LAST user message as the prompt to answer; the
    history slots in between as context the model "already said/heard".
    Empty content is dropped so we don't send blank turns to the API.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        for turn in history:
            content = (turn.content or "").strip()
            if not content:
                continue
            # OpenAI only accepts user/assistant/system/tool — anything else 400s.
            role = turn.role if turn.role in ("user", "assistant") else "user"
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


class OpenAIStreamingClient:
    """Wrapper around AsyncOpenAI chat.completions.

    Exposes two surfaces:
      * `stream()`   — server-streaming completion used for the final
                       user-facing answer. Yields TextDelta per chunk
                       and a terminal StreamDone with usage.
      * `complete()` — non-streaming completion used for short utility
                       calls (currently: history-aware query rewrite).
                       Returns a single CompletionResult.

    Each surface has its own model/temperature/max_tokens defaults:
    the chat path tends to want a capable model, while the rewrite path
    wants something fast and deterministic (temp=0).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        # Chat-completion defaults (used by stream()).
        default_model: str | None = None,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
        # Rewrite defaults (used by complete()).
        rewrite_model: str | None = None,
        rewrite_temperature: float | None = None,
        rewrite_max_tokens: int | None = None,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = AsyncOpenAI(api_key=api_key)

        # Defaults for the main chat-completion path.
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

        # Defaults for the query-rewrite path. Distinct env vars so ops can
        # point rewrite at a smaller/cheaper model without touching the
        # main answer model. Temp 0 → deterministic, single-line output.
        self._rewrite_model = rewrite_model or os.getenv(
            "LLM_REWRITE_MODEL", "gpt-4o-mini"
        )
        self._rewrite_temperature = (
            rewrite_temperature
            if rewrite_temperature is not None
            else float(os.getenv("LLM_REWRITE_TEMPERATURE", "0.0"))
        )
        self._rewrite_max_tokens = (
            rewrite_max_tokens
            if rewrite_max_tokens is not None
            else int(os.getenv("LLM_REWRITE_MAX_TOKENS", "256"))
        )
        log.info(
            "OpenAIStreamingClient ready chat_model=%r rewrite_model=%r temperature=%.2f max_tokens=%d",
            self._default_model,
            self._rewrite_model,
            self._default_temperature,
            self._default_max_tokens,
        )

    @property
    def rewrite_defaults(self) -> tuple[str, float, int]:
        """Exposed so the gRPC service layer can log/observe rewrite config."""
        return (self._rewrite_model, self._rewrite_temperature, self._rewrite_max_tokens)

    async def stream(
        self,
        *,
        system: str,
        user: str,
        history: Sequence[ChatTurn] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Streaming chat completion. Optional `history` is folded between
        the system prompt and the current user message — see _build_messages.
        """
        chosen_model = model or self._default_model
        chosen_temperature = (
            temperature if temperature and temperature > 0 else self._default_temperature
        )
        chosen_max_tokens = (
            max_tokens if max_tokens and max_tokens > 0 else self._default_max_tokens
        )

        log.info(
            "[openai] → chat.completions stream session_id=%r model=%r temp=%.2f max_tokens=%d sys_len=%d user_len=%d history_len=%d",
            session_id,
            chosen_model,
            chosen_temperature,
            chosen_max_tokens,
            len(system),
            len(user),
            len(history) if history else 0,
        )

        stream = await self._client.chat.completions.create(
            model=chosen_model,
            messages=_build_messages(system=system, history=history, user=user),
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
            stream=True,
            # include_usage=True asks OpenAI to send a final usage chunk
            # after the content stream ends — that's how we get token counts.
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

    async def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
    ) -> CompletionResult:
        """One-shot (non-streaming) chat completion.

        Used by the query-rewrite RPC: the caller wants a single string
        back, fast — streaming buys nothing here. Defaults fall back to
        the `rewrite_*` config rather than the main chat config.
        """
        chosen_model = model or self._rewrite_model
        # `temperature` may legitimately be 0.0; the truthy check is intentional —
        # 0.0 → use the rewrite default (also 0.0 by config). Callers that need a
        # non-zero override pass it explicitly.
        chosen_temperature = (
            temperature if temperature and temperature > 0 else self._rewrite_temperature
        )
        chosen_max_tokens = (
            max_tokens if max_tokens and max_tokens > 0 else self._rewrite_max_tokens
        )

        log.info(
            "[openai] → chat.completions (non-stream) session_id=%r model=%r temp=%.2f max_tokens=%d sys_len=%d user_len=%d",
            session_id,
            chosen_model,
            chosen_temperature,
            chosen_max_tokens,
            len(system),
            len(user),
        )

        response = await self._client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=chosen_temperature,
            max_tokens=chosen_max_tokens,
        )

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "").strip() if choice else ""
        finish_reason = (choice.finish_reason or "stop") if choice else "stop"
        usage = response.usage
        input_tokens = (usage.prompt_tokens or 0) if usage else 0
        output_tokens = (usage.completion_tokens or 0) if usage else 0

        log.info(
            "[openai] ← completion session_id=%r out_len=%d input_tokens=%d output_tokens=%d finish_reason=%r",
            session_id,
            len(text),
            input_tokens,
            output_tokens,
            finish_reason,
        )
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    async def close(self) -> None:
        await self._client.close()
