"""Apostrophe-aware segmentation for Turkish text.

Handles two distinct cases:
1. **Turkish proper names** — İstanbul'da, Ankara'ya
   → ROOT(İstanbul) + PUNCT(') + SUFFIX(da)
2. **Foreign stems with Turkish suffixes** — meeting'e, zoom'da
   → FOREIGN(meeting) + PUNCT(') + SUFFIX(e)

v2.1 changes:
- Rewritten ``is_turkish_base()`` with proper entity classification
- Consistent apostrophe PUNCT emission for all cases (Turkish + foreign)
- No Turkish I→ı lowering applied to non-Turkish entity surfaces

The decision between Turkish/foreign uses a priority-based classifier:
1. Contains digits → NOT Turkish proper (B2B, HTML5)
2. Known acronym → NOT Turkish proper (NATO, TBMM)
3. ALL CAPS + no Turkish chars + not in TDK → NOT Turkish (NVIDIA)
4. Has Turkish chars → Turkish
5. In proper nouns → Turkish
6. In TDK → Turkish
7. Mixed case → NOT Turkish (OpenAI, GitHub)
8. Short all-alpha lowercase → default Turkish
9. Otherwise → NOT Turkish
"""

from __future__ import annotations

import re

from ._acronym_table import ACRONYM_EXPANSIONS
from ._suffix_table import APOSTROPHE_SUFFIXES, SUFFIX_MAP
from .normalization import has_turkish_chars, is_all_caps_word, is_mixed_case, turkish_lower
from .resources import load_proper_nouns, load_tdk_words

# Matches word'suffix patterns (both ASCII and Unicode apostrophes)
_APO_RE = re.compile(
    r"([A-Za-zÇçĞğİıÖöŞşÜü0-9]{2,})['\u2019]([A-Za-zÇçĞğİıÖöŞşÜü]{1,6})\b"
)


def is_turkish_base(word: str) -> bool:
    """Return True if *word* should be treated as a Turkish word base.

    Used to decide whether ``word'suffix`` is a Turkish proper name
    (keep apostrophe as punctuation boundary) or a foreign word
    (merge into FOREIGN root + SUFFIX).

    Priority-based decision tree (checked in order):
    1. Contains digits (B2B, HTML5, 1990) → False (entity, not Turkish proper)
    2. ALL CAPS + in ACRONYM_EXPANSIONS → False (known acronym)
    3. ALL CAPS + no Turkish chars + not in TDK → False (foreign acronym like NVIDIA)
    4. Has Turkish-specific chars (ç,ğ,ı,ş,ö,ü) → True
    5. In proper nouns list → True
    6. In TDK dictionary → True
    7. Mixed case (OpenAI, GitHub) → False
    8. Short word (<4 chars), all alpha, not all-caps → True (default Turkish)
    9. Otherwise → False
    """
    # Rule 1: Words with digits are entities, not Turkish proper names
    if any(c.isdigit() for c in word):
        return False

    # Rule 2: Known acronyms in the expansion table
    if word.upper() in ACRONYM_EXPANSIONS:
        return False

    # Rule 3: ALL CAPS + no Turkish chars + not in TDK → foreign
    if is_all_caps_word(word):
        if not has_turkish_chars(word):
            wl = turkish_lower(word)
            tdk = load_tdk_words()
            if not (tdk and wl in tdk):
                return False

    # Rule 4: Turkish-specific characters are a strong signal
    # Use the original word (not lowered) to check for Turkish chars
    if has_turkish_chars(word):
        return True

    # Rule 5: Known proper nouns
    wl = turkish_lower(word)
    if wl in load_proper_nouns():
        return True

    # Rule 6: TDK dictionary
    tdk = load_tdk_words()
    if tdk and wl in tdk:
        return True

    # Rule 7: Mixed case → foreign brand / entity
    if is_mixed_case(word):
        return False

    # Rule 8: Very short all-alpha words default to Turkish
    alpha_only = all(c.isalpha() for c in word)
    if alpha_only and len(word) < 4 and not is_all_caps_word(word):
        return True

    # Rule 9: Otherwise → foreign
    return False


def split_apostrophe_words(
    text: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Process apostrophe patterns in *text*.

    For **foreign** stems followed by a Turkish suffix after apostrophe,
    replaces the apostrophe with a space so the word can later be
    segmented as FOREIGN ROOT + SUFFIX.

    For **Turkish** proper names (İstanbul'da), leaves the text
    unchanged — the apostrophe will be handled as punctuation by the
    word splitter.

    Returns:
        ``(modified_text, [(foreign_base_lower, suffix_lower), ...])``
    """
    foreign_splits: list[tuple[str, str]] = []

    def _repl(m: re.Match) -> str:
        base, suffix = m.group(1), m.group(2)

        if is_turkish_base(base):
            return m.group(0)  # Keep apostrophe for Turkish names

        sl = suffix.lower()
        if any(sl == s for s in APOSTROPHE_SUFFIXES):
            foreign_splits.append((turkish_lower(base), sl))
            return f"{base} {suffix}"  # Drop apostrophe → space

        return m.group(0)

    modified = _APO_RE.sub(_repl, text)
    return modified, foreign_splits


def build_apostrophe_tokens(
    word: str, suffix_str: str, *, is_foreign: bool
) -> list[dict[str, object]]:
    """Create token dicts for a word + apostrophe + suffix pattern.

    Args:
        word: The base word (before apostrophe).
        suffix_str: The suffix string (after apostrophe).
        is_foreign: Whether the base word is foreign.

    Returns:
        List of token dicts.
    """
    from .special_spans import split_apostrophe_suffixes

    suffix_pieces = split_apostrophe_suffixes(suffix_str)

    if is_foreign:
        # Foreign: FOREIGN(word) + PUNCT(') + SUFFIX chain
        tokens: list[dict[str, object]] = [
            {
                "token": f" {word}", "token_type": "FOREIGN", "morph_pos": 0,
                "_foreign": True,
            },
            {
                "token": "'", "token_type": "PUNCT", "morph_pos": 0,
                "_punct": True,
            },
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            tokens.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return tokens
    else:
        # Turkish: ROOT(word) + PUNCT(') + SUFFIX chain
        tokens = [
            {
                "token": f" {word}", "token_type": "ROOT", "morph_pos": 0,
            },
            {
                "token": "'", "token_type": "PUNCT", "morph_pos": 0,
                "_punct": True,
            },
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            tokens.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return tokens
