"""Conversational Query Rewriting Module.

Converts conversational, pronoun-heavy, or ambiguous follow-up questions
(e.g., 'What about the second one?' or 'Can you elaborate on that?')
into fully contextualized, standalone clinical queries before retrieval.
"""

from __future__ import annotations

import re
from typing import Any


_REWRITE_SYSTEM_PROMPT = """\
You are a clinical query reformulation assistant. Given a conversation history between a user and a clinical assistant, your job is to rewrite the user's latest follow-up question into a single, fully contextualized, standalone clinical query.

RULES:
1. Replace pronouns (it, this, that, the second one, option 2, the former, the latter, etc.) with explicit clinical terms, drug names, or age bands from the conversation history.
2. If the user input is already a complete, standalone question, return it unchanged.
3. Keep the query concise, precise, and clinically accurate.
4. Do NOT answer the question. Only output the rewritten standalone query string.
5. Return ONLY the rewritten query text. Do NOT include introductory phrases, explanation, quotes, or markdown.
"""


class ConversationalQueryRewriter:
    """Preprocesses user inputs with chat history to produce standalone queries."""

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0, max_turns: int = 4):
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns  # Sliding window of last 3-4 turns

    def _format_history(self, chat_history: list[dict[str, str]] | list[tuple[str, str]]) -> str:
        """Format the sliding window of history into text."""
        if not chat_history:
            return ""

        # Normalize history to dict format
        normalized: list[dict[str, str]] = []
        for item in chat_history:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append({"role": str(item[0]), "content": str(item[1])})

        # Apply sliding window of last max_turns (each turn is up to 2 messages: user + assistant)
        max_messages = self.max_turns * 2
        recent_history = normalized[-max_messages:]

        lines = []
        for msg in recent_history:
            role = "User" if msg.get("role") in ("user", "human") else "Assistant"
            lines.append(f"{role}: {msg.get('content', '').strip()}")

        return "\n".join(lines)

    def rewrite(
        self,
        query: str,
        chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
        skip_llm: bool = False,
    ) -> str:
        """Rewrite a conversational follow-up query into a standalone query.

        Args:
            query: The latest user input string.
            chat_history: List of previous turns.
            skip_llm: If True, use heuristic fallback reformulation.

        Returns:
            Standalone clinical query.
        """
        query_str = query.strip()
        if not query_str:
            return query_str

        if not chat_history:
            return query_str

        formatted_history = self._format_history(chat_history)
        if not formatted_history:
            return query_str

        if skip_llm:
            return self._heuristic_fallback(query_str, chat_history)

        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=self.model, temperature=self.temperature)
            user_prompt = (
                f"CONVERSATION HISTORY:\n{formatted_history}\n\n"
                f"LATEST USER QUERY:\n{query_str}\n\n"
                "STANDALONE REWRITTEN QUERY:"
            )
            messages = [
                ("system", _REWRITE_SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            rewritten = self._clean_llm_output(raw, query_str)
            return rewritten
        except Exception:
            return self._heuristic_fallback(query_str, chat_history)

    def _clean_llm_output(self, raw: str, original: str) -> str:
        """Clean raw LLM output to get pure query string."""
        cleaned = raw.strip()
        # Remove quotes if enclosed
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Remove "Standalone query:" prefixes if LLM included them
        cleaned = re.sub(r"^(standalone query|rewritten query|query):\s*", "", cleaned, flags=re.IGNORECASE)

        if not cleaned or len(cleaned.split()) < 2:
            return original
        return cleaned

    def _heuristic_fallback(
        self,
        query: str,
        chat_history: list[dict[str, str]] | list[tuple[str, str]],
    ) -> str:
        """Fallback rewrite using last user query context if LLM call is skipped or fails."""
        last_user_q = ""

        normalized: list[dict[str, str]] = []
        for item in chat_history:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append({"role": str(item[0]), "content": str(item[1])})

        for msg in reversed(normalized):
            role = msg.get("role")
            if role in ("user", "human") and not last_user_q:
                last_user_q = msg.get("content", "").strip()
                break

        followup_signals = re.compile(
            r"\b(second|first|third|that|this|it|them|elaborate|explain|more|side effects|details|above|latter|former)\b",
            re.IGNORECASE,
        )
        if followup_signals.search(query) and last_user_q:
            return f"{query} regarding {last_user_q}"

        return query


def rewrite_conversational_query(
    query: str,
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
    model: str = "llama3.2",
    skip_llm: bool = False,
) -> str:
    """Convenience function for query rewriting."""
    rewriter = ConversationalQueryRewriter(model=model)
    return rewriter.rewrite(query, chat_history=chat_history, skip_llm=skip_llm)
