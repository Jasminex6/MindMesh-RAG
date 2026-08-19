"""Terse Query Canonicalization and Multi-Turn Follow-Up Contextualization."""

from __future__ import annotations

import re
from .query_rewriter import rewrite_conversational_query


def canonicalize_query(query: str, chat_history: list[dict] | None = None) -> str:
    """Map terse noun-phrase queries or contextualize follow-up questions."""
    q_str = query.strip()
    if not q_str:
        return ""

    q_lower = q_str.lower()

    # 1. Multi-turn follow-up question resolution via ConversationalQueryRewriter
    if chat_history and len(chat_history) > 0:
        rewritten = rewrite_conversational_query(q_str, chat_history, skip_llm=False)
        if rewritten and rewritten != q_str:
            return rewritten

        # Terse follow-up pattern fallback
        follow_up_patterns = [
            r"\bhow is it\b", r"\bwhat are its\b", r"\bhow to treat it\b",
            r"\bwhat about children\b", r"\bhow do you treat it\b", r"\bwhat about\b"
        ]
        is_follow_up = any(re.search(pat, q_lower) for pat in follow_up_patterns) or (
            len(q_str.split()) <= 5 and "asthma" not in q_lower and not any(w in q_lower for w in ["what", "how", "why"])
        )

        if is_follow_up:
            last_user_query = ""
            for msg in reversed(chat_history):
                if msg.get("role") == "user" and msg.get("content"):
                    last_user_query = msg["content"]
                    break

            if last_user_query:
                topic = "paediatric asthma" if ("paediatric" in last_user_query.lower() or "child" in last_user_query.lower()) else "asthma"
                if "treatment" in q_lower or "treated" in q_lower or "manage" in q_lower:
                    return f"How is {topic} treated?"
                elif "symptom" in q_lower:
                    return f"What are the symptoms of {topic}?"
                elif "diagnos" in q_lower:
                    return f"How is {topic} diagnosed?"
                else:
                    return f"{q_str} regarding {topic}"

    # 2. Terse query mapping
    terse_map = {
        "asthma symptoms": "What are the symptoms of asthma?",
        "paediatric asthma symptoms": "What are the symptoms of paediatric asthma?",
        "pediatric asthma symptoms": "What are the symptoms of paediatric asthma?",
        "asthma treatment": "How is asthma treated?",
        "paediatric asthma treatment": "How is paediatric asthma treated?",
        "pediatric asthma treatment": "How is paediatric asthma treated?",
        "asthma triggers": "What triggers asthma?",
        "asthma diagnosis": "How is asthma diagnosed?",
        "asthma management": "How is asthma managed?",
    }

    return terse_map.get(q_lower, q_str)
