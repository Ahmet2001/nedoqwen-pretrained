"""Text normalization utilities for Turkish text.

Handles:
- Unicode NFC normalization (position-preserving)
- Turkish-aware lowercasing (İ→i, I→ı) — used on per-word copies, never on main text
- Turkish character detection
- Per-word ALL CAPS detection (non-destructive replacement for old detect_all_caps)

Design note: normalization is intentionally NON-DESTRUCTIVE.  The public
token surface must match the original input.  All lowercasing is done on
internal analysis copies, never on the text that produces output surfaces.
"""

from __future__ import annotations

import unicodedata

# Turkish-specific characters — presence indicates a Turkish word
TR_CHARS: frozenset[str] = frozenset("çğışöüÇĞİŞÖÜ")

# Turkish uppercase letters (including special İ)
_TR_UPPER: frozenset[str] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZÇĞİÖŞÜ")


# ASCII-only Turkish fallback substitutions used for noisy / diacriticless text.
ASCII_TR_SUBS: dict[str, str] = {
    "c": "ç", "g": "ğ", "i": "ı", "o": "ö", "s": "ş", "u": "ü",
}


def turkish_lower(s: str) -> str:
    """Turkish-aware lowercase: İ→i, I→ı, then standard ``str.lower()``.

    Standard Python ``str.lower()`` maps both I and İ to 'i', which is
    wrong for Turkish where I→ı and İ→i.

    This function is used ONLY on internal analysis copies of words,
    never on the main text stream or output surface forms.
    """
    return s.replace("İ", "i").replace("I", "ı").lower()


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization only.

    NFC is position-preserving for Turkish text: it converts combining
    character sequences into their precomposed forms (e.g., c + cedilla →
    ç) without changing character positions for already-precomposed text.

    NO whitespace collapse is performed — this preserves character positions
    for offset tracking and ensures token surfaces can be sliced directly
    from the original input.
    """
    return unicodedata.normalize("NFC", text)


def has_turkish_chars(word: str) -> bool:
    """Return True if *word* contains Turkish-specific characters (ç,ğ,ı,ş,ö,ü)."""
    return any(c in TR_CHARS for c in word)


def is_all_caps_word(word: str) -> bool:
    """Return True if *word* is an ALL CAPS word (≥2 uppercase letters, no lowercase).

    This is a per-word check used in the segmentation engine to detect
    acronyms and ALL-CAPS words WITHOUT destructively modifying the text.

    Words containing digits (like HTML5, B2B) are considered ALL CAPS
    if all their alphabetic characters are uppercase.
    """
    alpha_chars = [c for c in word if c.isalpha()]
    if len(alpha_chars) < 2:
        return False
    return all(c in _TR_UPPER or c.isupper() for c in alpha_chars)


def is_mixed_case(word: str) -> bool:
    """Return True if *word* has mixed case (e.g., OpenAI, GitHub).

    A word is mixed-case if it contains both uppercase and lowercase
    letters AND doesn't fit the pattern of a normal Turkish word
    (which would be either all-lower or first-letter-capitalized).
    """
    alpha_chars = [c for c in word if c.isalpha()]
    if len(alpha_chars) < 2:
        return False
    has_upper = any(c.isupper() or c in _TR_UPPER for c in alpha_chars)
    has_lower = any(c.islower() for c in alpha_chars)
    if not (has_upper and has_lower):
        return False
    # Normal capitalization (first-letter-upper, rest lower) is NOT mixed case
    # Mixed case = uppercase letters appear after position 0
    return any(
        (c.isupper() or c in _TR_UPPER) for c in alpha_chars[1:]
    )

def find_ascii_turkish_variant(word: str, lexicon: set[str], *, max_variants: int = 256) -> str | None:
    """Return the best Turkish-diacritic variant of *word* found in *lexicon*.

    The selector is conservative: it prefers variants with the lowest weighted
    edit cost from the ASCII surface. Consonant-diacritic repairs are treated as
    slightly cheaper than vowel-diacritic repairs so ``cok`` prefers ``çok``
    over ``cök`` / ``çök``.
    """
    if not word or not word.isascii() or word in lexicon:
        return None

    positions = [i for i, ch in enumerate(word) if ch in ASCII_TR_SUBS]
    if not positions:
        return None

    positions = positions[:8]
    chars = list(word)
    variants_checked = 0
    matches: list[tuple[int, str]] = []
    weights = {"c": 1, "g": 1, "s": 1, "i": 2, "o": 2, "u": 2}

    def _search(idx: int, cost: int) -> None:
        nonlocal variants_checked
        if variants_checked >= max_variants:
            return
        if idx >= len(positions):
            variants_checked += 1
            candidate = "".join(chars)
            if candidate != word and candidate in lexicon:
                matches.append((cost, candidate))
            return

        pos = positions[idx]
        original = chars[pos]

        # Keep original character.
        _search(idx + 1, cost)

        # Try Turkish-diacritic repair.
        chars[pos] = ASCII_TR_SUBS[original]
        _search(idx + 1, cost + weights.get(original, 1))
        chars[pos] = original

    _search(0, 0)
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches[0][1]

