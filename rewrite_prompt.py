"""Prompt template for the history-aware query-rewrite LLM call.

The rewrite step turns a (possibly context-dependent) user message into a
self-contained query suitable for vector retrieval. The LLM is instructed
to merge any relevant prior-turn context into the rewritten query, but to
return ONLY the query text — no preamble, quotes, or explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from llm.v1 import llm_pb2


REWRITE_SYSTEM_PROMPT = """\
You rewrite the user's latest message into a single, self-contained search query
for a retrieval system. The retrieval system has no access to prior turns of
the conversation, so the query must stand on its own.

Rules:
1. Resolve pronouns and demonstratives ("it", "that", "this", "they",
   "the one you mentioned") using the prior conversation.
2. Carry over the topic implied by earlier turns when the user's latest
   message is a follow-up, refinement, comparison, or disambiguation.
3. If the latest message is already self-contained, return it unchanged
   (or with only minimal cleanup).
4. If the latest message is greeting / small-talk / not a real question,
   return it verbatim.
5. Do NOT answer the question. Do NOT add explanation, quotes, or labels.
6. Output ONLY the rewritten query as a single line of plain text.
"""


@dataclass(frozen=True)
class _RenderedRewrite:
    system: str
    user: str


_ROLE_LABEL = {
    llm_pb2.MESSAGE_ROLE_USER: "User",
    llm_pb2.MESSAGE_ROLE_ASSISTANT: "Assistant",
}


def _render_history(history: Iterable[llm_pb2.ChatMessage]) -> str:
    lines: list[str] = []
    for msg in history:
        label = _ROLE_LABEL.get(msg.role, "User")
        content = (msg.content or "").strip()
        if not content:
            continue
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def render_rewrite_prompt(
    history: Sequence[llm_pb2.ChatMessage],
    user_message: str,
) -> _RenderedRewrite:
    history_block = _render_history(history) or "(no prior turns)"
    user = (
        "CONVERSATION SO FAR:\n"
        f"{history_block}\n\n"
        "LATEST USER MESSAGE:\n"
        f"{user_message.strip()}\n\n"
        "Rewritten standalone query:"
    )
    return _RenderedRewrite(system=REWRITE_SYSTEM_PROMPT, user=user)
