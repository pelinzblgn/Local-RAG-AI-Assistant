from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.agent_models import (
    AgentDecision,
    AgentIntent,
)
from src.agent_tools import ToolRegistry


@dataclass(frozen=True)
class IntentRule:
    """One deterministic routing rule."""

    intent: AgentIntent
    tool_name: str
    patterns: tuple[str, ...]
    confidence: float


class AgentDecisionEngine:
    """
    Deterministic first-stage decision engine.

    Explicit operational requests are routed only to
    registered capabilities. Ordinary questions fall
    back to local knowledge search.

    Operational requests never fall back to knowledge
    search when their required capability is unavailable.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        if not isinstance(
            registry,
            ToolRegistry,
        ):
            raise TypeError(
                "registry must be a ToolRegistry."
            )

        self._registry = registry

        self._rules = (
            IntentRule(
                intent=AgentIntent.KNOWLEDGE_SYNC,
                tool_name="knowledge_sync",
                patterns=(
                    r"\bsenkronize\b",
                    r"\bsenkronizasyon\b",
                    r"\bsync\b",
                    r"\bguncelle\b",
                    r"\bguncelleme\b",
                    r"\byenile\b",
                    r"\brefresh\b",
                ),
                confidence=0.99,
            ),
            IntentRule(
                intent=AgentIntent.KNOWLEDGE_STATUS,
                tool_name="knowledge_status",
                patterns=(
                    r"\bbilgi tabaninda ne var\b",
                    r"\bbilgi tabani durumu\b",
                    r"\bknowledge base status\b",
                    r"\bkac belge\b",
                    r"\bkac dokuman\b",
                    r"\bkac kaynak\b",
                    r"\bindeks durumu\b",
                    r"\bindex status\b",
                ),
                confidence=0.98,
            ),
            IntentRule(
                intent=AgentIntent.CONVERSATION_CLEAR,
                tool_name="conversation_clear",
                patterns=(
                    r"\bgecmisi temizle\b",
                    r"\bkonusma gecmisini temizle\b",
                    r"\bhafizayi temizle\b",
                    r"\bclear history\b",
                    r"\bclear conversation\b",
                    r"\bclear memory\b",
                ),
                confidence=0.99,
            ),
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize user input for deterministic routing.

        Turkish characters, capitalization,
        punctuation and repeated whitespace are
        normalized before rule matching.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Agent input must be a string."
            )

        clean_text = text.strip()

        if not clean_text:
            raise ValueError(
                "Agent input cannot be empty."
            )

        clean_text = clean_text.casefold()

        replacements = str.maketrans(
            {
                "ı": "i",
                "ğ": "g",
                "ü": "u",
                "ş": "s",
                "ö": "o",
                "ç": "c",
            }
        )

        clean_text = clean_text.translate(
            replacements
        )

        clean_text = unicodedata.normalize(
            "NFKD",
            clean_text,
        )

        clean_text = "".join(
            character
            for character in clean_text
            if not unicodedata.combining(
                character
            )
        )

        clean_text = re.sub(
            r"[^\w\s]",
            " ",
            clean_text,
        )

        return re.sub(
            r"\s+",
            " ",
            clean_text,
        ).strip()

    def _tool_available(
        self,
        tool_name: str,
    ) -> bool:
        """Return whether a capability is registered."""

        return self._registry.contains(
            tool_name
        )

    def decide(
        self,
        user_input: str,
    ) -> AgentDecision:
        """
        Determine the safest supported action for
        one user request.

        Explicit operational requests never fall back
        to knowledge search when their required
        capability is unavailable.
        """

        normalized = self._normalize(
            user_input
        )

        for rule in self._rules:
            matched = any(
                re.search(
                    pattern,
                    normalized,
                )
                for pattern in rule.patterns
            )

            if not matched:
                continue

            if self._tool_available(
                rule.tool_name
            ):
                return AgentDecision(
                    intent=rule.intent,
                    tool_name=rule.tool_name,
                    reason=(
                        "The request matched a supported "
                        f"{rule.intent.value} operation."
                    ),
                    confidence=rule.confidence,
                )

            return AgentDecision(
                intent=AgentIntent.UNKNOWN,
                tool_name=None,
                reason=(
                    "The request matched the "
                    f"{rule.intent.value} operation, "
                    "but the required capability "
                    f"'{rule.tool_name}' is not registered."
                ),
                confidence=1.0,
            )

        if self._tool_available(
            "knowledge_search"
        ):
            return AgentDecision(
                intent=AgentIntent.KNOWLEDGE_QUERY,
                tool_name="knowledge_search",
                reason=(
                    "No explicit operational command was "
                    "detected, so the request will be "
                    "handled as a local knowledge query."
                ),
                confidence=0.85,
            )

        return AgentDecision(
            intent=AgentIntent.UNKNOWN,
            tool_name=None,
            reason=(
                "No supported registered tool can handle "
                "the request."
            ),
            confidence=0.0,
        )