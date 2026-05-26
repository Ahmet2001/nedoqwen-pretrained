"""Special span detection: URLs, emails, numbers, dates, mentions, hashtags, emojis, acronyms.

Detects non-textual spans in the input text **before** the word-level
segmentation runs, so they are never mistakenly split by suffix
stripping.  Returns a sorted, non-overlapping list of spans.

v2.1 changes:
- EMAIL detection (before MENTION to prevent @ stealing)
- URL trailing punctuation stripping
- NUM+UNIT compound detection (15kg → NUM + UNIT)
- Hyphenated entity detection (COVID-19, GPT-4, Wi-Fi)
- Operates on original (non-lowered) text for correct acronym detection
"""

from __future__ import annotations

import re

from ._acronym_table import ACRONYM_EXPANSIONS
from ._suffix_table import APOSTROPHE_SUFFIXES
from .normalization import turkish_lower
from .resources import load_proper_nouns, load_tdk_words

# ── Static vocabulary sets ───────────────────────────────────────────────────

MONTH_NAMES: frozenset[str] = frozenset({
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
})

UNITS: frozenset[str] = frozenset({
    "km", "m", "cm", "mm", "nm",
    "kg", "g", "mg", "ton",
    "sn", "dk", "sa", "ms",
    "tl", "usd", "eur", "gbp",
    "kb", "mb", "gb", "tb", "pb",
    "ml", "mcg", "meq", "iu", "mmhg", "mosm",
    "hz", "mhz", "ghz", "watt", "kw", "mw", "kcal", "cal",
})

ROMAN_NUMERALS: frozenset[str] = frozenset({
    "i", "ii", "iii", "iv", "vi", "vii", "viii", "ix",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
})

# ── Regex patterns ───────────────────────────────────────────────────────────

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Email: must come before MENTION to prevent @-stealing
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

MENTION_RE = re.compile(r"@[\w\u00C0-\u024F]+")
HASHTAG_RE = re.compile(r"#[\w\u00C0-\u024F]+")

_SUFFIX_ALT = "|".join(re.escape(s) for s in APOSTROPHE_SUFFIXES)

# Number + apostrophe + Turkish suffix(es)
NUM_APOSTROPHE_RE = re.compile(
    r"\d+(?:[.:,]\d+)*['\u2019](?:" + _SUFFIX_ALT + r")+\b",
    re.IGNORECASE,
)

DATE_RE = re.compile(
    r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}"
    r"|\d{4}[./\-]\d{1,2}[./\-]\d{1,2}"
)
CURRENCY_RE = re.compile(r"[$€£¥₺₽]\d+[\.,]?\d*|\d+[\.,]?\d*[$€£¥₺₽]")
NUMBER_RE = re.compile(
    r"%\d+[\.,]?\d*"
    r"|\d{1,3}(?:\.\d{3})+"     # thousands (1.000.000)
    r"|\d+[\.,]\d+"             # decimal
    r"|\d+%"
    r"|\d+/\d+"
)
TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")

# Number + unit compound (15kg, 5ms) — unit list built from UNITS set
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
NUM_UNIT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)(" + _UNIT_ALT + r")\b",
    re.IGNORECASE,
)

PLAIN_NUM_RE = re.compile(r"\b\d+\b")

# Acronyms: standalone uppercase 2+ letters (optionally + digits)
# Operates on ORIGINAL text (not lowered), so İSTANBUL, ODTÜ, TBMM match correctly
ACRONYM_RE = re.compile(
    r"\b[A-ZÇĞİÖŞÜ]{2,}[0-9]*\b"
    r"|\b[A-ZÇĞİÖŞÜ][0-9]+\b"
)

# Acronym + apostrophe + Turkish suffix(es)
ACRONYM_APOSTROPHE_RE = re.compile(
    r"\b(?:[A-ZÇĞİÖŞÜ]{2,}[0-9]*|[A-ZÇĞİÖŞÜ][0-9]+)['\u2019](?:"
    + _SUFFIX_ALT + r")+\b"
)

# Hyphenated entity + apostrophe + suffix: COVID-19'dan, GPT-4'ün, Llama-3'te
# Pattern: WORD-WORD or WORD-NUM where at least one part has uppercase
HYPHENATED_ENTITY_APO_RE = re.compile(
    r"\b[A-Za-zÇçĞğİıÖöŞşÜü]+"              # first part (letters)
    r"(?:-[A-Za-zÇçĞğİıÖöŞşÜü0-9]+)+"       # one or more hyphen+part
    r"['\u2019](?:" + _SUFFIX_ALT + r")+\b",  # apostrophe + suffix(es)
    re.IGNORECASE,
)

# Standalone hyphenated entity (no apostrophe): COVID-19, Wi-Fi, GPT-4
HYPHENATED_ENTITY_RE = re.compile(
    r"\b[A-Za-zÇçĞğİıÖöŞşÜü]+"
    r"(?:-[A-Za-zÇçĞğİıÖöŞşÜü0-9]+)+\b"
)

# v2.1.1: Technical spans: paths, versions, language names, e-devlet
PATH_RE = re.compile(r"(?:/[a-zA-Z0-9._-]+){2,}|(?:\b[a-zA-Z]:\\[a-zA-Z0-9._\-\\]+)")
VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+)+(?:[a-zA-Z0-9._-]*)\b")
CPP_RE = re.compile(r"C\+\+|C#|F#", re.IGNORECASE)
E_DEVLET_RE = re.compile(r"\be-devlet\b", re.IGNORECASE)
E_DEVLET_APO_RE = re.compile(r"\be-devlet['\u2019](?:" + _SUFFIX_ALT + r")+\b", re.IGNORECASE)

TEXT_EMOJI_RE = re.compile(r"[:;=]-?[\)\(\]\[dDpPoO3]|<3")
UNICODE_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)

# ── Trailing punctuation chars to strip from URLs ────────────────────────────
_URL_TRAIL_STRIP: frozenset[str] = frozenset(".,;:!?)")

# Priority order: earlier entries win when spans overlap.
_SPAN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (URL_RE,                    "URL"),
    (EMAIL_RE,                  "EMAIL"),
    (CPP_RE,                    "TECH"),
    (HASHTAG_RE,                "HASHTAG"),
    (DATE_RE,                   "DATE"),
    (VERSION_RE,                "TECH"),
    (E_DEVLET_APO_RE,           "ROOT_APO"),
    (E_DEVLET_RE,               "ROOT"),
    (MENTION_RE,                "MENTION"),
    (PATH_RE,                   "TECH"),
    (CURRENCY_RE,               "UNIT"),
    (NUM_APOSTROPHE_RE,         "NUM_APO"),
    (HYPHENATED_ENTITY_APO_RE,  "HYPHENATED_APO"),
    (ACRONYM_APOSTROPHE_RE,     "ACRONYM_APO"),
    (ACRONYM_RE,                "ACRONYM"),
    (NUMBER_RE,                 "NUM"),
    (TIME_RE,                   "NUM"),
    (NUM_UNIT_RE,               "NUM_UNIT"),
    (PLAIN_NUM_RE,              "NUM"),
    (UNICODE_EMOJI_RE,          "EMOJI"),
    (TEXT_EMOJI_RE,              "EMOJI"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_hyphenated_entity(text: str) -> bool:
    """Return True if a hyphenated span looks like a named entity (not plain Turkish).

    Entity indicators: uppercase letters, digits, mixed case.
    Plain Turkish: all-lowercase, no digits (çok-güzel, mavi-beyaz).
    """
    parts = text.split("-")
    if not parts:
        return False
    # Must have at least one part with uppercase or digit
    has_entity_signal = any(
        any(c.isupper() for c in p) or any(c.isdigit() for c in p)
        for p in parts
    )
    return has_entity_signal


# ── Acronym vs Turkish word disambiguation ───────────────────────────────────

def _is_known_turkish_word(word_upper: str) -> bool:
    """Return True if *word_upper* (ALL CAPS) is actually a Turkish word.
    
    v2.1.2: Stricter check for all-caps Turkish verbs vs acronyms.
    """
    if word_upper in ACRONYM_EXPANSIONS:
        return False
    
    wl = turkish_lower(word_upper)
    tdk = load_tdk_words()
    proper = load_proper_nouns()

    if wl in tdk or wl in proper:
        # Standalone TDK match: usually not an acronym unless very short (e.g. AB)
        if len(wl) >= 3:
            return True

    # Shallow morphological parse for common inflections (YAPTI, DEDİ)
    from ._suffix_table import SUFFIX_ENTRIES
    for sfx_surf, _ in SUFFIX_ENTRIES:
        if len(sfx_surf) < 2:
            continue
        if wl.endswith(sfx_surf):
            rem = wl[: -len(sfx_surf)]
            if len(rem) >= 2:
                # Check if rem is a valid verb stem
                if rem in tdk or (f"{rem}mak" in tdk) or (f"{rem}mek" in tdk):
                    return True
    return False


def _trim_url_trailing_punct(
    start: int, end: int, text: str
) -> tuple[int, int]:
    """Strip trailing punctuation characters from a URL span.

    Characters stripped: . , ; : ! ? )
    Also handles paired brackets: if char before start is '(' and URL
    ends with ')', strip the ')'.
    """
    while end > start:
        last_char = text[end - 1]
        if last_char in _URL_TRAIL_STRIP:
            end -= 1
        else:
            break
    return start, end


# ── Public API ───────────────────────────────────────────────────────────────

def find_special_spans(text: str) -> list[tuple[int, int, str, str]]:
    """Find all special-token spans in *text*.

    Returns a sorted, non-overlapping list of
    ``(start, end, token_type, original_text)``.
    """
    candidates: list[tuple[int, int, str, str]] = []
    for pattern, ttype in _SPAN_PATTERNS:
        for m in pattern.finditer(text):
            original = m.group(0)
            span_start = m.start()
            span_end = m.end()

            # Acronym filtering: skip if it's actually a common Turkish word
            if ttype in ("ACRONYM", "ACRONYM_APO"):
                if ttype == "ACRONYM_APO":
                    apo = original.find("'")
                    if apo == -1:
                        apo = original.find("\u2019")
                    acr_base = original[:apo]
                else:
                    acr_base = original
                if _is_known_turkish_word(acr_base):
                    continue

            # Hyphenated entity filtering: skip if it looks like plain Turkish
            if ttype in ("HYPHENATED_APO", "HYPHENATED_ENTITY"):
                if ttype == "HYPHENATED_APO":
                    apo = original.find("'")
                    if apo == -1:
                        apo = original.find("\u2019")
                    base = original[:apo]
                else:
                    base = original
                if not _is_hyphenated_entity(base):
                    continue

            # URL trailing punctuation trimming
            if ttype == "URL":
                span_start, span_end = _trim_url_trailing_punct(
                    span_start, span_end, text
                )
                original = text[span_start:span_end]

            candidates.append((span_start, span_end, ttype, original))

    # Sort by start position, then prefer longer match
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Greedy non-overlapping selection
    result: list[tuple[int, int, str, str]] = []
    last_end = 0
    for s, e, t, o in candidates:
        if s >= last_end:
            result.append((s, e, t, o))
            last_end = e
    return result


def split_apostrophe_suffixes(suffix_str: str) -> list[tuple[str, str]]:
    """Split a suffix string (after apostrophe) into individual suffix pieces.

    Returns a list of ``(surface_form, label)`` tuples.
    """
    from ._suffix_table import SUFFIX_MAP  # avoid circular at module level

    pieces: list[tuple[str, str]] = []
    remaining = suffix_str.lower()
    while remaining:
        matched = False
        for s in APOSTROPHE_SUFFIXES:
            if remaining.startswith(s):
                label = SUFFIX_MAP.get(s, "-SFX")
                pieces.append((s, label))
                remaining = remaining[len(s):]
                matched = True
                break
        if not matched:
            # Unrecognised remainder → emit as a single suffix chunk
            pieces.append((remaining, "-SFX"))
            break
    return pieces


def make_special_tokens(
    span_type: str, original: str
) -> list[dict[str, object]]:
    """Create token dict(s) for a matched special span.

    ``NUM_APO``, ``ACRONYM_APO``, ``HYPHENATED_APO``, and ``NUM_UNIT``
    spans are split into base + SUFFIX / UNIT tokens.
    """
    # ── Number + apostrophe + suffix (3'te, 1990'larda) ──────────────────
    if span_type == "NUM_APO":
        apo_pos = original.find("'")
        if apo_pos == -1:
            apo_pos = original.find("\u2019")
        actual_apo = original[apo_pos]
        num_part = original[:apo_pos]
        suffix_pieces = split_apostrophe_suffixes(original[apo_pos + 1:])
        result: list[dict[str, object]] = [
            {"token": num_part, "token_type": "NUM", "morph_pos": 0, "_num": True},
            {"token": actual_apo, "token_type": "PUNCT", "morph_pos": 0, "_punct": True},
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            result.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return result

    # ── Acronym + apostrophe + suffix (NATO'nun, HTML5'ten) ──────────────
    if span_type == "ACRONYM_APO":
        apo_pos = original.find("'")
        if apo_pos == -1:
            apo_pos = original.find("\u2019")
        actual_apo = original[apo_pos]
        acr_part = original[:apo_pos]
        suffix_pieces = split_apostrophe_suffixes(original[apo_pos + 1:])
        expansion = ACRONYM_EXPANSIONS.get(acr_part.upper())
        # Also check without trailing digits for expansions (e.g. HTML5 → HTML)
        if not expansion:
            base_alpha = acr_part.rstrip("0123456789")
            if base_alpha:
                expansion = ACRONYM_EXPANSIONS.get(base_alpha.upper())
        meta: dict[str, object] = {"_acronym": True}
        if expansion:
            meta["_expansion"] = expansion
            meta["_known_acronym"] = True
        result = [
            {"token": acr_part, "token_type": "ACRONYM", "morph_pos": 0, **meta},
            {"token": actual_apo, "token_type": "PUNCT", "morph_pos": 0, "_punct": True},
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            result.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return result

    # ── e-devlet with suffixes (e-devlet'te) ─────────────────────────────
    if span_type == "ROOT_APO":
        apo_pos = max(original.find("'"), original.find("\u2019"))
        actual_apo = original[apo_pos]
        root_part = original[:apo_pos]
        suffix_pieces = split_apostrophe_suffixes(original[apo_pos + 1:])
        result = [
            {"token": root_part, "token_type": "ROOT", "morph_pos": 0, "_root": True},
            {"token": actual_apo, "token_type": "PUNCT", "morph_pos": 0, "_punct": True},
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            result.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return result

    # ── Hyphenated entity + apostrophe + suffix (COVID-19'dan, GPT-4'ün) ─
    if span_type == "HYPHENATED_APO":
        apo_pos = original.find("'")
        if apo_pos == -1:
            apo_pos = original.find("\u2019")
        actual_apo = original[apo_pos]
        entity_part = original[:apo_pos]
        suffix_pieces = split_apostrophe_suffixes(original[apo_pos + 1:])
        result = [
            {"token": entity_part, "token_type": "ACRONYM", "morph_pos": 0,
             "_acronym": True, "_hyphenated": True},
            {"token": actual_apo, "token_type": "PUNCT", "morph_pos": 0, "_punct": True},
        ]
        for idx, (surf, label) in enumerate(suffix_pieces, start=1):
            result.append({
                "token": surf, "token_type": "SUFFIX", "morph_pos": idx,
                "_apo_suffix": True, "_suffix_label": label,
            })
        return result

    # ── Standalone hyphenated entity (COVID-19, Wi-Fi) ───────────────────
    if span_type == "HYPHENATED_ENTITY":
        return [{
            "token": original, "token_type": "ACRONYM", "morph_pos": 0,
            "_acronym": True, "_hyphenated": True,
        }]

    # ── Number + unit compound (15kg, 5ms) ───────────────────────────────
    if span_type == "NUM_UNIT":
        # original is the full match; we need to split num and unit
        # The regex has groups: group(1)=number, group(2)=unit
        # But since we stored original as m.group(0), re-parse it
        m = NUM_UNIT_RE.match(original)
        if m:
            num_part = m.group(1)
            unit_part = m.group(2)
        else:
            # Fallback: try to split digits from letters
            i = 0
            while i < len(original) and (original[i].isdigit() or original[i] in ".,"):
                i += 1
            num_part = original[:i]
            unit_part = original[i:]
        return [
            {"token": num_part, "token_type": "NUM", "morph_pos": 0, "_num": True},
            {"token": unit_part, "token_type": "UNIT", "morph_pos": 0, "_unit": True},
        ]

    # ── Plain acronym (HTML5, GPT) ──────────────────────────────────────
    if span_type == "ACRONYM":
        expansion = ACRONYM_EXPANSIONS.get(original.upper())
        if not expansion:
            base_alpha = original.rstrip("0123456789")
            if base_alpha:
                expansion = ACRONYM_EXPANSIONS.get(base_alpha.upper())
        meta = {"_acronym": True}
        if expansion:
            meta["_expansion"] = expansion
            meta["_known_acronym"] = True
        return [{"token": original, "token_type": "ACRONYM", "morph_pos": 0, **meta}]

    # ── Email ────────────────────────────────────────────────────────────
    if span_type == "EMAIL":
        return [{"token": original, "token_type": "EMAIL", "morph_pos": 0, "_email": True}]

    # ── Everything else (NUM, DATE, URL, MENTION, HASHTAG, EMOJI, UNIT, TECH) ──
    return [{
        "token": original,
        "token_type": span_type,
        "morph_pos": 0,
        f"_{span_type.lower()}": True,
    }]


def reclassify_numbers_in_tokens(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    """Post-pass: catch remaining numbers / units missed by span detection."""
    result: list[dict[str, object]] = []
    for tok in tokens:
        tt = tok["token_type"]
        if tt not in ("ROOT", "FOREIGN"):
            result.append(tok)
            continue

        raw = str(tok["token"]).strip()

        if NUMBER_RE.fullmatch(raw):
            result.append({**tok, "token_type": "NUM", "_num": True})
        elif raw.lower() in UNITS:
            result.append({**tok, "token_type": "UNIT", "_unit": True})
        elif raw.lower() in ROMAN_NUMERALS:
            result.append({**tok, "token_type": "NUM", "_roman": True})
        elif raw.lower() in MONTH_NAMES:
            result.append({**tok, "token_type": "ROOT", "_month": True})
        else:
            result.append(tok)

    return result
