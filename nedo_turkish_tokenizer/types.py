"""Core type definitions for NedoTurkishTokenizer.

Defines the Token dataclass, SegmentationCandidate for the candidate-based
segmentation engine, token type constants, and punctuation character sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Token type constants ──────────────────────────────────────────────────────

ROOT = "ROOT"
SUFFIX = "SUFFIX"
FOREIGN = "FOREIGN"
PUNCT = "PUNCT"
NUM = "NUM"
DATE = "DATE"
UNIT = "UNIT"
URL = "URL"
MENTION = "MENTION"
HASHTAG = "HASHTAG"
EMOJI = "EMOJI"
ACRONYM = "ACRONYM"
EMAIL = "EMAIL"
TECH = "TECH"

# Special token types that represent non-textual entities
SPECIAL_TYPES: frozenset[str] = frozenset(
    {NUM, DATE, UNIT, URL, MENTION, HASHTAG, EMOJI, ACRONYM, EMAIL, TECH}
)

# All recognized token types
ALL_TYPES: frozenset[str] = frozenset(
    {ROOT, SUFFIX, FOREIGN, PUNCT, NUM, DATE, UNIT, URL, MENTION, HASHTAG, EMOJI, ACRONYM, EMAIL, TECH}
)

# ── Punctuation character set ────────────────────────────────────────────────

PUNCT_CHARS: frozenset[str] = frozenset(
    "'?.,;:!-\u2013\u2014()[]{}\"`/\\|@#$%^&*+=<>~"
    "\u2019\u2018\u201c\u201d\u2032\u00ab\u00bb\u2039\u203a"
    "\u2022\u2026\u00b7\u00b0\u00b1\u00d7\u00f7"
)

# Digits — used alongside PUNCT_CHARS for pure-punctuation detection
_DIGITS: frozenset[str] = frozenset("0123456789")


def is_punct_token(text: str) -> bool:
    """Return True if *text* consists entirely of punctuation / digit characters."""
    stripped = text.strip()
    if not stripped:
        return False
    return all(
        c in PUNCT_CHARS or c in _DIGITS or (ord(c) > 0x02FF and not c.isalpha())
        for c in stripped
    )


# ── Token dataclass ──────────────────────────────────────────────────────────


@dataclass
class Token:
    """Internal token representation.

    *text* uses the leading-space convention: a space prefix indicates
    that this token starts a new word.  Suffixes within a word have
    no leading space.

    The *metadata* dict carries optional annotation fields (all prefixed
    with ``_``), for example ``_caps``, ``_foreign``, ``_canonical``.
    """

    text: str
    token_type: str
    morph_pos: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the public API dict format."""
        result: dict[str, Any] = {
            "token": self.text,
            "token_type": self.token_type,
            "morph_pos": self.morph_pos,
        }
        result.update(self.metadata)
        return result


# ── Segmentation candidate ───────────────────────────────────────────────────


@dataclass
class SegmentationCandidate:
    """One possible way to segment a word into tokens.

    The candidate-generation engine produces multiple candidates per word,
    then the selection step picks the highest-scoring one.

    *source* is a short human-readable tag describing the strategy that
    produced this candidate (e.g. ``"tdk_root"``, ``"suffix_chain"``,
    ``"foreign"``).
    """

    tokens: list[Token]
    score: float
    source: str
