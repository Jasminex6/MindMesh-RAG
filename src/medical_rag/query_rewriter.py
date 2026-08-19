"""Conversational Query Rewriter module.

Reformulates implicit, pronoun-heavy, or contextual follow-up questions into
complete, self-contained standalone clinical queries before vector retrieval.
"""

from __future__ import annotations

import re
from typing import Any


_REWRITE_SYSTEM_PROMPT = """\
You are an expert clinical query rephraser.
Given a conversation history between a user and an AI clinical assistant, and a follow-up user query, your task is to rewrite the follow-up query into a clear, complete, self-contained standalone clinical question.

RULES:
1. Replace pronouns and vague references (e.g. "it", "they", "that", "the second one", "side effects", "what about treatment") with the specific clinical subjects (condition, drug, age group, severity) established in the conversation history.
2. Preserve all clinical context (e.g. paediatric vs. adult, drug names, dosage thresholds).
3. If the query is already self-contained and clear on its own, return it unchanged.
4. Do NOT answer the question. Output ONLY the single rewritten standalone clinical question without quotes or explanations.
"""


def _normalize_history(
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None,
    max_messages: int = 8,
) -> list[dict[str, str]]:
    """Convert tuple or dict history into a standardized list of dicts and take sliding window."""
    if not chat_history:
        return []

    standardized: list[dict[str, str]] = []
    for turn in chat_history:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            standardized.append({"role": role, "content": content})
        elif isinstance(turn, (tuple, list)) and len(turn) >= 2:
            standardized.append({"role": "user", "content": str(turn[0])})
            standardized.append({"role": "assistant", "content": str(turn[1])})

    return standardized[-max_messages:]


def _heuristic_fallback(
    query: str,
    chat_history: list[dict[str, str]],
) -> str:
    """Deterministic fallback when LLM is unavailable or skip_llm=True.

    Extracts recent topic anchors from past turns if the query is an implicit follow-up.
    """
    stripped = query.strip()
    if not chat_history:
        return stripped

    # Search past user turns for clinical context words
    past_topics: list[str] = []
    user_msgs = [m for m in chat_history if m.get("role") == "user"]
    search_list = reversed(user_msgs) if user_msgs else reversed(chat_history)
    for msg in search_list:
        text = msg.get("content", "")
        # Remove common question headers to isolate the subject
        cleaned = re.sub(r"^(what|how|why|when|is|can|should|could)\s+(are|is|do|does|i|we|a|the)?\s*", "", text, flags=re.IGNORECASE).strip()
        if cleaned and len(cleaned) > 3:
            past_topics.append(cleaned)
            break

    if not past_topics:
        return stripped

    anchor = past_topics[0]
    q_lower = stripped.lower()

    # If query contains pronouns or is a brief fragment, append anchor
    pronoun_patterns = [r"\bit\b", r"\bthat\b", r"\bthis\b", r"\bthese\b", r"\bthose\b", r"\bthey\b", r"\bsecond\s+one\b"]
    is_fragment = len(stripped.split()) <= 4 or any(re.search(p, q_lower) for p in pronoun_patterns)

    if is_fragment:
        # Check if query starts with "what about" or "how about"
        if re.match(r"^(what|how)\s+about\b", q_lower):
            sub = re.sub(r"^(what|how)\s+about\s*", "", stripped, flags=re.IGNORECASE).strip("? ")
            return f"{anchor} - {sub}"
        # Substitute "it" / "that" if present
        if re.search(r"\b(it|that)\b", q_lower):
            substituted = re.sub(r"\b(it|that)\b", anchor, stripped, flags=re.IGNORECASE)
            return substituted

        return f"{anchor} {stripped}"

    return stripped


class ConversationalQueryRewriter:
    """Service encapsulating conversational query reformulation into standalone queries."""

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.0,
        max_turns: int = 4,
    ):
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns

    def rewrite(
        self,
        query: str,
        chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
        skip_llm: bool = False,
    ) -> str:
        """Rewrite a user query into a standalone clinical question using chat history."""
        stripped_query = query.strip()
        if not stripped_query:
            return ""

        norm_history = _normalize_history(chat_history, max_messages=self.max_turns * 2)
        if not norm_history:
            return stripped_query

        if skip_llm:
            return _heuristic_fallback(stripped_query, norm_history)

        # Build prompt for LLM rewriter
        formatted_turns = []
        for msg in norm_history:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            formatted_turns.append(f"{role_label}: {msg.get('content', '').strip()}")

        history_block = "\n".join(formatted_turns)
        user_prompt = (
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            f"FOLLOW-UP USER QUERY:\n{stripped_query}\n\n"
            "REWRITTEN STANDALONE CLINICAL QUESTION:"
        )

        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=self.model, temperature=self.temperature, timeout=1.5)
            messages = [
                ("system", _REWRITE_SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
            response = llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            cleaned = raw_text.strip().strip('"').strip("'")

            # Basic sanity check on LLM output
            if cleaned and len(cleaned) >= 5 and not cleaned.lower().startswith("sorry"):
                return cleaned
            return _heuristic_fallback(stripped_query, norm_history)

        except Exception as e:
            print(f"[QueryRewriter] Ollama call failed ({e}). Using heuristic fallback.")
            return _heuristic_fallback(stripped_query, norm_history)


def rewrite_conversational_query(
    query: str,
    chat_history: list[dict[str, str]] | list[tuple[str, str]] | None = None,
    model: str = "llama3.2",
    skip_llm: bool = False,
) -> str:
    """Convenience functional wrapper for query rewriting."""
    rewriter = ConversationalQueryRewriter(model=model)
    return rewriter.rewrite(query=query, chat_history=chat_history, skip_llm=skip_llm)
