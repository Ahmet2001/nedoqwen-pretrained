"""Turkish suffix pattern table (260+ entries).

Maps surface-form suffixes to morphological labels.  Used by the
segmentation engine for candidate generation (suffix stripping) and by
the post-annotation layer for ``_suffix_label`` metadata.

Suffixes are sorted longest-first at module load time so that the
candidate generator always tries the most specific match first.

Design note: some surface forms are ambiguous (e.g. "in" can be GEN or
2SG).  This table assigns a single canonical label per surface form —
the most common interpretation in written Turkish.  The candidate scoring
system resolves segmentation ambiguity via root validation, not via
suffix-label disambiguation.
"""

from __future__ import annotations

# ── Raw suffix → label mapping ───────────────────────────────────────────────
# Organised by morphological category for readability.

SUFFIX_MAP: dict[str, str] = {
    # ── Plural + case (REMOVED to encourage finer segmentation) ─────────────
    # "leri": "-PL+ACC",   "ları": "-PL+ACC",
    # "lere": "-PL+DAT",   "lara": "-PL+DAT",
    # "lerin": "-PL+GEN",  "ların": "-PL+GEN",
    # "lerde": "-PL+LOC",  "larda": "-PL+LOC",
    # "lerden": "-PL+ABL",  "lardan": "-PL+ABL",
    # "lerle": "-PL+INS",  "larla": "-PL+INS",
    "lerce": "-PL+EQU",  "larca": "-PL+EQU",
    # ── Loanword / derivational ──────────────────────────────────────────
    "yon": "-YON",   "iyon": "-YON",   "asyon": "-YON",   "izasyon": "-YON",
    # ── Adjective derivation ─────────────────────────────────────────────
    "sal": "-ADJ.TR",  "sel": "-ADJ.TR",
    # ── 1st/2nd plural possessive ────────────────────────────────────────
    "imiz": "-P1PL",  "ımız": "-P1PL",  "umuz": "-P1PL",  "ümüz": "-P1PL",
    "iniz": "-P2PL",  "ınız": "-P2PL",  "unuz": "-P2PL",  "ünüz": "-P2PL",
    # ── Participial / Nominalizers ──────────────────────────────────────
    "dığı": "-PART+P3", "diği": "-PART+P3", "duğu": "-PART+P3", "düğü": "-PART+P3",
    "tığı": "-PART+P3", "tiği": "-PART+P3", "tuğu": "-PART+P3", "tüğü": "-PART+P3",
    "dık": "-PART", "dik": "-PART", "duk": "-PART", "dük": "-PART",
    "dığ": "-PART.SOFT", "diğ": "-PART.SOFT", "duğ": "-PART.SOFT", "düğ": "-PART.SOFT",
    "tık": "-PART", "tik": "-PART", "tuk": "-PART", "tük": "-PART",
    "ınca": "-ADV.TIME", "ince": "-ADV.TIME", "unca": "-ADV.TIME", "ünce": "-ADV.TIME",
    "arak": "-ADV.CONV", "erek": "-ADV.CONV",
    "alı": "-ADV.SINCE", "eli": "-ADV.SINCE",
    # ── Question / Particle ──────────────────────────────────────────────
    "mıdır": "-Q+EPIS", "midir": "-Q+EPIS", "mudur": "-Q+EPIS", "müdür": "-Q+EPIS",
    "mışsa": "-EVID+COND", "mişse": "-EVID+COND", "muşsa": "-EVID+COND", "müşse": "-EVID+COND",
    "sa": "-COND",  "se": "-COND",
    "ki": "-REL",
    "da": "-CONJ",  "de": "-CONJ",
    # ── Aspect / Tense ───────────────────────────────────────────────────
    "nır": "-PASS.AOR", "nir": "-PASS.AOR", "nur": "-PASS.AOR", "nür": "-PASS.AOR",
    "ar": "-AOR",  "er": "-AOR",  "ır": "-AOR",  "ir": "-AOR",  "ur": "-AOR",  "ür": "-AOR",  "r": "-AOR",
    "iyor": "-PROG",  "ıyor": "-PROG",  "uyor": "-PROG",  "üyor": "-PROG",
    "yor": "-PROG",
    "makta": "-PROG.CONT", "mekte": "-PROG.CONT",
    "dı": "-PST",   "di": "-PST",   "du": "-PST",   "dü": "-PST",
    "tı": "-PST",   "ti": "-PST",   "tu": "-PST",   "tü": "-PST",
    "mış": "-EVID",  "miş": "-EVID",  "muş": "-EVID",  "müş": "-EVID",
    # ── Negation ─────────────────────────────────────────────────────────
    "ma": "-NEG",  "me": "-NEG",
    "mı": "-NEG.PROG", "mi": "-NEG.PROG", "mu": "-NEG.PROG", "mü": "-NEG.PROG",
    "ama": "-ABIL+NEG", "eme": "-ABIL+NEG",
    "lama": "-VN+NEG",  "leme": "-VN+NEG",
    "maya": "-NEG.INF",
    # ── Future tense ────────────────────────────────────────────────────
    "ecek": "-FUT",  "acak": "-FUT",
    "yecek": "-FUT",  "yacak": "-FUT",
    "eceğ": "-FUT.SOFT", "acağ": "-FUT.SOFT", "yeceğ": "-FUT.SOFT", "yacağ": "-FUT.SOFT",
    "eceği": "-FUT+P3", "acağı": "-FUT+P3",
    "ecekti": "-FUT+PST", "acaktı": "-FUT+PST",
    # ── Negative aorist ─────────────────────────────────────────────────
    "mez": "-NEG.AOR",  "maz": "-NEG.AOR",
    # ── While-doing ─────────────────────────────────────────────────────
    "mekte": "-VN+LOC",  "makta": "-VN+LOC",
    # ── Abilitative ──────────────────────────────────────────────────────
    "bil": "-ABIL",
    # ── Necessitative ────────────────────────────────────────────────────
    "malı": "-NECES",  "meli": "-NECES",
    # ── Infinitive ───────────────────────────────────────────────────────
    "mak": "-INF",  "mek": "-INF",
    # ── -ken (while/when) ────────────────────────────────────────────────
    "ken": "-WHEN",
    # ── Converb ──────────────────────────────────────────────────────────
    "arak": "-CONV",  "erek": "-CONV",
    # ── With / without ───────────────────────────────────────────────────
    "lı": "-WITH",   "li": "-WITH",   "lu": "-WITH",   "lü": "-WITH",
    "sız": "-WITHOUT", "siz": "-WITHOUT", "suz": "-WITHOUT", "süz": "-WITHOUT",
    # ── Agentive ─────────────────────────────────────────────────────────
    "cı": "-AGT",  "ci": "-AGT",  "cu": "-AGT",  "cü": "-AGT",
    "çı": "-AGT",  "çi": "-AGT",  "çu": "-AGT",  "çü": "-AGT",
    # ── Abstract noun ────────────────────────────────────────────────────
    "lık": "-ABSTR",  "lik": "-ABSTR",  "luk": "-ABSTR",  "lük": "-ABSTR",
    "lığ": "-ABSTR",  "liğ": "-ABSTR",
    # ── Optative 1pl ─────────────────────────────────────────────────────
    "elim": "-OPT1PL",  "alım": "-OPT1PL",
    # ── Person suffixes ──────────────────────────────────────────────────
    "ım": "-1SG",  "im": "-1SG",  "um": "-1SG",  "üm": "-1SG",
    "m": "-1SG",   "n": "-2SG",   "k": "-1PL",
    "sın": "-2SG",  "sin": "-2SG",  "sun": "-2SG",  "sün": "-2SG",
    "iz": "-1PL",  "ız": "-1PL",  "uz": "-1PL",  "üz": "-1PL",
    "nız": "-2PL",  "niz": "-2PL",  "nuz": "-2PL",  "nüz": "-2PL",
    # ── Question ─────────────────────────────────────────────────────────
    "mı": "-Q",  "mi": "-Q",  "mu": "-Q",  "mü": "-Q",
    # ── Accusative ───────────────────────────────────────────────────────
    "yı": "-ACC",  "yi": "-ACC",  "yu": "-ACC",  "yü": "-ACC",
    "ı": "-ACC",   "i": "-ACC",   "u": "-ACC",   "ü": "-ACC",
    # ── Relatives (v2.1.2: forced grouping) ──────────────────────────────
    "daki": "-REL+LOC", "deki": "-REL+LOC", "taki": "-REL+LOC", "teki": "-REL+LOC",
    "ndaki": "-REL+LOC", "ndeki": "-REL+LOC",
    # ── Dative ───────────────────────────────────────────────────────────
    "ya": "-DAT",  "ye": "-DAT",
    "a": "-DAT",   "e": "-DAT",
    # ── Ablative ─────────────────────────────────────────────────────────
    "dan": "-ABL",  "den": "-ABL",  "tan": "-ABL",  "ten": "-ABL",
    # ── Locative ─────────────────────────────────────────────────────────
    "da": "-LOC",  "de": "-LOC",  "ta": "-LOC",  "te": "-LOC",
    # ── Plural ───────────────────────────────────────────────────────────
    "lar": "-PL",  "ler": "-PL",
    # ── 3sg possessive ───────────────────────────────────────────────────
    "sı": "-P3",  "si": "-P3",  "su": "-P3",  "sü": "-P3",
    # ── Genitive ─────────────────────────────────────────────────────────
    "nin": "-GEN",  "nın": "-GEN",  "nun": "-GEN",  "nün": "-GEN",
    "ın": "-GEN",   "in": "-GEN",   "un": "-GEN",   "ün": "-GEN",
    # ── Instrumental ─────────────────────────────────────────────────────
    "le": "-INS",  "la": "-INS",
    # ── Equative ─────────────────────────────────────────────────────────
    "ce": "-EQU",  "ca": "-EQU",  "çe": "-EQU",  "ça": "-EQU",
    # ── Frequent BPE-origin suffixes ─────────────────────────────────────
    "iril": "-PASS.SFX",
    "yan": "-PART.ACT", "ren": "-PART.ACT",
    "ri": "-PL.SFX",
}

# Suffixes that are too short / ambiguous for aggressive stripping.
SHORT_AMBIGUOUS_SUFFIXES: frozenset[str] = frozenset(
    {"a", "e", "ı", "i", "u", "ü"}
)

# Pre-sorted list: (surface_form, label) ordered longest-first.
SUFFIX_ENTRIES: list[tuple[str, str]] = sorted(
    SUFFIX_MAP.items(), key=lambda x: len(x[0]), reverse=True
)


# ── Turkish suffixes that can follow an apostrophe ───────────────────────────
APOSTROPHE_SUFFIXES: list[str] = sorted(
    [
        "nın", "nin", "nun", "nün", "dan", "den", "tan", "ten",
        "da", "de", "ta", "te", "ya", "ye", "nda", "nde",
        "yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü",
        "lar", "ler", "lara", "lere", "ları", "leri",
        "ım", "im", "um", "üm", "ın", "in", "un", "ün",
        "mız", "miz", "muz", "müz", "nız", "niz", "nuz", "nüz",
        "ki", "li", "lı", "lu", "lü", "sız", "siz", "suz", "süz",
        "daki", "deki",
        "lık", "lik", "luk", "lük",
        "a", "e", "ı", "i", "u", "ü",
    ],
    key=len,
    reverse=True,
)
