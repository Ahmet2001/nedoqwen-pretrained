"""Tokenization engine — orchestrates the full pipeline.

This is the central pipeline that ties together all modules:
1. Text normalization (Unicode NFC only — non-destructive)
2. Special span extraction (URLs, emails, numbers, dates, acronyms, emojis)
3. Word-level segmentation with candidate generation/selection
4. Post-annotation (allomorph labels, compound info, acronym expansion)
5. Number/unit reclassification safety net
6. morph_pos computation and internal whitespace stripping

v2.1 changes:
- Removed destructive detect_all_caps() — per-word caps detection instead
- Removed whitespace collapse — preserves original positions
- Pipeline operates on original text; analysis uses turkish_lower() per-word
- Token surfaces come from original input, never from mutilated copies
"""

from __future__ import annotations

from ._domain_vocab import ALL_DOMAIN_ROOTS
from .morphology import annotate_acronyms, annotate_canonical, annotate_compounds
from .normalization import normalize_text
from .resources import load_tdk_words
from .segmentation import segment_word, split_into_words
from .special_spans import find_special_spans, make_special_tokens, reclassify_numbers_in_tokens


class TokenizationEngine:
    """Core tokenization engine.

    Stateless after initialisation: loads TDK and domain vocabulary once,
    then processes texts through a deterministic pipeline.

    This class is NOT the public API.  Use ``NedoTurkishTokenizer``
    instead, which delegates to this engine.
    """

    def __init__(self) -> None:
        self._tdk: set[str] = load_tdk_words()
        self._domain_roots: frozenset[str] = ALL_DOMAIN_ROOTS

    def tokenize(self, text: str) -> list[dict[str, object]]:
        """Run the full tokenization pipeline on *text*.

        Returns a list of token dicts, each with at minimum:
        ``token``, ``token_type``, ``morph_pos``.

        Token surfaces preserve the original casing and form from *text*.
        """
        if not text:
            return []

        # ── 1. Normalize (NFC only — position-preserving for already precomposed) ────
        text = normalize_text(text)

        # ── 2. Special span extraction ───────────────────────────────────
        spans = find_special_spans(text)

        tokens: list[dict[str, object]] = []
        pos = 0

        for start, end, span_type, original in spans:
            # Tokenize normal text before this special span
            if pos < start:
                gap = text[pos:start]
                seg_tokens = self._tokenize_segment(gap)
                tokens.extend(seg_tokens)

            # Insert special tokens directly
            tokens.extend(make_special_tokens(span_type, original))
            pos = end

        # Tokenize remaining text after last special span
        if pos < len(text):
            gap = text[pos:]
            seg_tokens = self._tokenize_segment(gap)
            tokens.extend(seg_tokens)

        # ── 4. Post-annotation passes ────────────────────────────────────
        tokens = reclassify_numbers_in_tokens(tokens)
        tokens = annotate_canonical(tokens)
        tokens = annotate_compounds(tokens)
        tokens = annotate_acronyms(tokens)

        # ── 5. Finalize morph_pos ────────────────────────────────────────
        tokens = _compute_morph_pos(tokens)

        return tokens

    def _tokenize_segment(self, segment: str) -> list[dict[str, object]]:
        """Tokenize a plain-text segment (no special spans), preserving spaces."""
        import re
        # Split by whitespace but keep the delimiters
        parts = re.split(r"(\s+)", segment)
        tokens: list[dict[str, object]] = []

        for part in parts:
            if not part:
                continue
            if part.isspace():
                tokens.append({
                    "token": part,
                    "token_type": "SPACE",
                    "morph_pos": 0,
                    "_space": True
                })
            else:
                word_tokens = segment_word(
                    part, self._tdk, self._domain_roots,
                )
                tokens.extend(word_tokens)

        return tokens


# ── Helper: compute morph_pos across the full token stream ───────────────────

def _compute_morph_pos(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recompute ``morph_pos`` consistently across the token stream.

    Rules:
    - Word-initial tokens (ROOT, FOREIGN, PUNCT, Special) → morph_pos = 0
    - SUFFIX tokens increment the position counter relative to the last word-start
    - Apostrophe suffixes continue from the previous word
    - SPACE tokens are ignored for morphological counting
    """
    result: list[dict[str, object]] = []
    word_pos = 0

    for tok in tokens:
        token_type = str(tok["token_type"])

        # SPACE tokens don't reset or participate in morph_pos chains
        if token_type == "SPACE":
            result.append({**tok, "morph_pos": 0})
            continue

        # Suffixes after apostrophe continue the chain
        is_apo_suffix = tok.get("_apo_suffix")

        if is_apo_suffix:
            word_pos += 1
            morph_pos = word_pos
        elif token_type == "SUFFIX":
            word_pos += 1
            morph_pos = word_pos
        else:
            # Word-start (ROOT, PUNCT, Special)
            word_pos = 0
            morph_pos = 0

        result.append({**tok, "morph_pos": morph_pos})

    return result
