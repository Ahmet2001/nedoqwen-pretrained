"""Static resource loading for the tokenizer.

Loads bundled data files (TDK dictionary, proper nouns) from the package
``data/`` directory.  All resources are loaded lazily on first access and
cached in module-level globals.

**No network access.  No runtime downloads.  Fully offline.**

The TDK dictionary contains infinitive verb forms (e.g. "gelmek") but the
tokenizer needs bare verb stems (e.g. "gel") for suffix stripping.  This
module automatically derives verb stems from infinitives at load time.
"""

from __future__ import annotations

from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


# ── TDK Word List ────────────────────────────────────────────────────────────

_TDK_WORDS: set[str] | None = None


def _derive_verb_stems(raw_words: set[str]) -> set[str]:
    """Derive bare verb stems from TDK infinitive entries.

    TDK lists verbs as infinitives ("gelmek", "bakmak").  The tokenizer
    needs bare stems ("gel", "bak") for suffix stripping.

    This function strips "-mak"/"-mek" from infinitives and adds the
    resulting stems to the word set.  Only stems of 2+ characters are
    added to avoid spurious short matches.
    """
    derived: set[str] = set()
    for word in raw_words:
        if word.endswith("mak") and len(word) > 4:
            stem = word[:-3]
            if len(stem) >= 2:
                derived.add(stem)
        elif word.endswith("mek") and len(word) > 4:
            stem = word[:-3]
            if len(stem) >= 2:
                derived.add(stem)
    return derived


def load_tdk_words() -> set[str]:
    """Load the TDK (Türk Dil Kurumu) word list from the bundled data file.

    Returns a set of lowercase Turkish words including:
    - Original dictionary entries (nouns, adjectives, adverbs, infinitives)
    - Derived verb stems (stripped -mak/-mek from infinitives)

    Used for:
    - Root validation during suffix stripping (is the remainder a real word?)
    - Foreign word detection (word absent from TDK → likely foreign)
    - Turkish-base detection for apostrophe handling
    """
    global _TDK_WORDS
    if _TDK_WORDS is not None:
        return _TDK_WORDS

    tdk_path = _DATA_DIR / "tdk_words.txt"
    if tdk_path.exists():
        raw_words = {
            line.strip().lower()
            for line in tdk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        # Derive verb stems from infinitives (gelmek→gel, bakmak→bak)
        stems = _derive_verb_stems(raw_words)
        _TDK_WORDS = raw_words | stems
    else:
        _TDK_WORDS = set()

    return _TDK_WORDS


# ── Proper Nouns ─────────────────────────────────────────────────────────────

_PROPER_NOUNS: set[str] | None = None


def load_proper_nouns() -> set[str]:
    """Load Turkish proper nouns (cities, regions, names) from bundled data.

    Used in apostrophe handling to distinguish Turkish proper names
    (İstanbul'da → keep as Turkish ROOT) from foreign words
    (meeting'e → mark as FOREIGN ROOT).
    """
    global _PROPER_NOUNS
    if _PROPER_NOUNS is not None:
        return _PROPER_NOUNS

    path = _DATA_DIR / "turkish_proper_nouns.txt"
    if path.exists():
        _PROPER_NOUNS = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    else:
        _PROPER_NOUNS = set()

    return _PROPER_NOUNS
