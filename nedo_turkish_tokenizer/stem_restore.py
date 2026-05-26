"""Turkish stem restoration: consonant softening reversal & vowel insertion.

v2.1.2+: Handles verb narrowing (daralma) correctly for 1-char and multi-char stems.
"""

from __future__ import annotations

_CONSONANT_REVERSE: dict[str, str] = {
    "b": "p", "d": "t", "c": "ç", "g": "k", "ğ": "k",
}

_VOWELS: frozenset[str] = frozenset("aeıioöuü")
_VOWEL_INSERT_MAP: dict[str, str] = {
    "a": "ı", "ı": "ı", "o": "u", "u": "u",
    "e": "i", "i": "i", "ö": "ü", "ü": "ü",
}

def _last_vowel(word: str) -> str | None:
    for ch in reversed(word):
        if ch in _VOWELS: return ch
    return None

def _is_vowel(ch: str) -> bool:
    return ch in _VOWELS

def restore_consonant(remainder: str, tdk: set[str]) -> str | None:
    if not remainder or remainder in tdk: return None
    last = remainder[-1]
    if last not in _CONSONANT_REVERSE: return None
    candidate = remainder[:-1] + _CONSONANT_REVERSE[last]
    return candidate if candidate in tdk else None

def restore_vowel_drop(remainder: str, tdk: set[str]) -> str | None:
    if not remainder or len(remainder) < 2 or _is_vowel(remainder[-1]) or _is_vowel(remainder[-2]):
        return None
    lv = _last_vowel(remainder)
    if not lv: return None
    candidate = remainder[:-1] + _VOWEL_INSERT_MAP.get(lv, "i") + remainder[-1]
    return candidate if candidate in tdk else None

def restore_noun_stem(remainder: str, tdk: set[str]) -> str | None:
    res = restore_consonant(remainder, tdk)
    if res: return res
    res = restore_vowel_drop(remainder, tdk)
    if res: return res
    if remainder and remainder[-1] in _CONSONANT_REVERSE:
        res = restore_vowel_drop(remainder[:-1] + _CONSONANT_REVERSE[remainder[-1]], tdk)
        if res: return res
    return None

def restore_verb_stem(surface_stem: str, tdk: set[str]) -> str | None:
    """Restore verb stem (narrowing then softening)."""
    if not surface_stem: return None
    
    # 1. Verb narrowing (daralma) - MUST BE FIRST (some narrow stems don't end in softenables)
    if len(surface_stem) >= 1:
        # Special cases for ye and de
        if surface_stem in {"y", "yi", "yu", "yü"}: return "ye"
        if surface_stem in {"d", "di", "du", "dü"}: return "de"
        
        last_v = surface_stem[-1]
        narrow_candidate = None
        if last_v in "ıua": narrow_candidate = surface_stem[:-1] + "a"
        elif last_v in "iüe": narrow_candidate = surface_stem[:-1] + "e"
        
        if narrow_candidate:
            if narrow_candidate in tdk or (narrow_candidate + "mak" in tdk) or (narrow_candidate + "mek" in tdk):
                return narrow_candidate

    if surface_stem in tdk: return None

    # 2. Consonant softening reversal
    last = surface_stem[-1]
    if last in _CONSONANT_REVERSE:
        cand = surface_stem[:-1] + _CONSONANT_REVERSE[last]
        # Softening reversal for verbs restricted to primitive set + causal 't' doesn't soften.
        if last == "d" and cand not in {"git", "et", "tat", "güt", "dit"}:
            pass
        elif cand in tdk or (cand + "mak" in tdk) or (cand + "mek" in tdk):
            return cand

    return None
