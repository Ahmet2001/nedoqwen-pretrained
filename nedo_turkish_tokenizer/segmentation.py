"""Word-level segmentation with candidate generation and selection.

v2.1.22: stability pass.
Balanced morphological priority with robustness fixes for noisy ASCII Turkish.
"""

from __future__ import annotations

import re
from typing import Any

from ._domain_vocab import ALL_DOMAIN_ROOTS
from ._suffix_table import (
    SHORT_AMBIGUOUS_SUFFIXES,
    SUFFIX_ENTRIES,
    SUFFIX_MAP,
)
from .normalization import find_ascii_turkish_variant, has_turkish_chars, is_all_caps_word, turkish_lower
from .resources import load_proper_nouns, load_tdk_words
from .stem_restore import restore_noun_stem, restore_verb_stem
from .types import PUNCT_CHARS, SegmentationCandidate, Token, is_punct_token

# ── Scoring constants ────────────────────────────────────────────────────────
_TDK_BONUS = 80.0          
_DOMAIN_BONUS = 50.0        
_SUFFIX_BONUS = -3.0       
_ROOT_LEN_WEIGHT = 10.0     
_WHOLE_WORD_BONUS = 50.0   
_FOREIGN_BASE = 5.0       
_UNKNOWN_BASE = 1.0        
_SHORT_ROOT_PENALTY = 20.0 
_MIN_ROOT_LEN = 2         
_MAX_SUFFIX_DEPTH = 10    

_UNKNOWN_LEN_PENALTY = 5.0   
_RESTORED_ROOT_BONUS = 80.0  
_PROGRESSIVE_BONUS = 60.0    
_SUFFIX_LEN_WEIGHT = 4.0     
_VERB_CHAIN_BONUS = 40.0     
_TOKEN_COUNT_PENALTY = -5.0  
_NONVERBAL_FINITE_PENALTY = 120.0
_ASCII_WHOLE_WORD_BONUS = 25.0
_SHORT_AMBIGUOUS_PERSON_PENALTY = 90.0
_PREFERRED_RESTORED_NOUN_BONUS = 45.0

# ── Known-intact words ───────────────────────────────────────────────────────
KNOWN_INTACT: frozenset[str] = frozenset({
    "nasıl", "neden", "niçin", "belki", "dünya", "toplantı",
    "diye", "niye", "nice", "yeni", "beri", "geri", "dolu",
    "yani", "araba", "cuma", "dedi", "yedi",
})

# High-priority verb primitives for irregular paradigms
_PRIMITIVE_VERBS: frozenset[str] = frozenset({
    "de", "ye", "et", "git", "ol", "yap", "bul", "gel", "bak", "gör", "al", "ver"
})

_BUFFER_VOWELS: frozenset[str] = frozenset("ıiuü")

_LEADING_PUNCT_RE = re.compile(r"^([^\w]+)")
_TRAILING_PUNCT_RE = re.compile(r"([^\w]+)$")

_UNVOICED_CHARS: frozenset[str] = frozenset("pçtkfşsh")

_FINITE_VERBAL_SUFFIXES: frozenset[str] = frozenset({
    "-PST", "-EVID", "-PROG", "-FUT", "-NEG", "-NEG.PROG", "-COND", "-AOR", "-PASS.AOR"
})

_PREFERRED_RESTORED_NOUN_SURFACES: frozenset[str] = frozenset({"kanad"})


def _resolve_tdk_form(surface: str, tdk: set[str]) -> str | None:
    """Return the exact or ASCII-normalized lexicon form for *surface*."""
    if surface in tdk:
        return surface
    return find_ascii_turkish_variant(surface, tdk)


def _root_token(surface: str, lemma: str, *, is_caps: bool = False, restored: bool = False) -> Token:
    metadata: dict[str, object] = {"_root_lemma": lemma}
    if is_caps:
        metadata["_caps"] = True
    if restored:
        metadata["_restored"] = True
        metadata["_surface_root"] = surface
    elif lemma != surface:
        metadata["_ascii_normalized"] = True
        metadata["_surface_root"] = surface
    else:
        metadata["_surface_root"] = surface
    return Token(text=surface if not restored else lemma, token_type="ROOT", metadata=metadata)


def _split_punctuation(word: str) -> list[tuple[str, str]]:
    if not word: return []
    parts: list[tuple[str, str]] = []
    if is_punct_token(word): return [(word, "PUNCT")]
    lead_m = _LEADING_PUNCT_RE.match(word)
    if lead_m:
        for ch in lead_m.group(1): parts.append((ch, "PUNCT"))
        word = word[lead_m.end():]
    trail_m = _TRAILING_PUNCT_RE.search(word)
    trailing: list[tuple[str, str]] = []
    if trail_m:
        for ch in trail_m.group(1): trailing.append((ch, "PUNCT"))
        word = word[:trail_m.start()]
    if word: parts.append((word, "WORD"))
    parts.extend(trailing)
    return parts


def split_into_words(text: str) -> list[str]:
    return text.split()


def _is_verb(rt: str, tdk: set[str]) -> bool:
    return (rt + "mak" in tdk) or (rt + "mek" in tdk) or (rt in {"ye", "de", "et", "git", "tat"})

def _generate_suffix_candidates(
    word_lower: str,
    tdk: set[str],
    domain_roots: frozenset[str],
    depth: int = 0,
    allow_rootless: bool = False,
    following_label: str | None = None,
) -> list[SegmentationCandidate]:
    if depth >= _MAX_SUFFIX_DEPTH or not word_lower:
        return []

    candidates: list[SegmentationCandidate] = []

    for suffix_surface, suffix_label in SUFFIX_ENTRIES:
        if not word_lower.endswith(suffix_surface):
            continue

        remainder = word_lower[: -len(suffix_surface)]
        
        # ── Suffix Ordering Constraints ──
        # Root + Aspect + Person (Correct)
        # Root + Person + Aspect (Invalid)
        # ── Suffix Ordering Constraints ──
        if following_label:
            # Person suffixes (-1SG, -2PL, etc) MUST come after Aspect/Tense.
            # Right-to-left: if we already found an Aspect suffix to the right,
            # we CANNOT match a Person suffix to its left.
            is_person = any(suffix_label.endswith(p) for p in ("SG", "PL", "P1PL", "P2PL"))
            following_is_aspect = following_label.startswith(("-PROG", "-PST", "-EVID", "-FUT", "-NEG", "-ABIL"))
            if is_person and following_is_aspect:
                continue
            # Aorist/passive-aorist cannot sit immediately to the left of progressive.
            if following_label == "-PROG" and suffix_label in {"-AOR", "-PASS.AOR", "-ACC"}:
                continue

        is_narrow_root = remainder in {"y", "d", "yi", "di", "yu", "du", "yü", "dü"}
        if len(remainder) < _MIN_ROOT_LEN and not is_narrow_root and remainder != "":
            continue
        
        # Guardrail: 1-char narrow roots (d, y) only allowed for specific verbal suffixes
        if len(remainder) == 1 and is_narrow_root:
            if not suffix_label.startswith(("-PROG", "-PST", "-EVID", "-FUT", "-NEG")):
                continue

        suffix_token = Token(text=suffix_surface, token_type="SUFFIX", metadata={"_suffix_label": suffix_label})

        if remainder == "" and allow_rootless:
             candidates.append(SegmentationCandidate(tokens=[suffix_token], score=(len(suffix_surface) * _SUFFIX_LEN_WEIGHT) + _SUFFIX_BONUS, source="suffix_only"))
             continue

        matched_root = _resolve_tdk_form(remainder, tdk)
        root_in_tdk = matched_root is not None
        root_in_domain = remainder in domain_roots
        restored_root: str | None = None
        restored_kind: str | None = None
        root_score = len(remainder) * _ROOT_LEN_WEIGHT

        if root_in_tdk:
            root_score += _TDK_BONUS
            if matched_root != remainder:
                root_score += _ASCII_WHOLE_WORD_BONUS
        elif root_in_domain:
            root_score += _DOMAIN_BONUS
        else:
            restored_root = restore_noun_stem(remainder, tdk)
            if restored_root is not None:
                restored_kind = "noun"
            else:
                restored_root = restore_verb_stem(remainder, tdk)
                if restored_root is not None:
                    restored_kind = "verb"
            if restored_root:
                root_score += _RESTORED_ROOT_BONUS
            else:
                root_score += _UNKNOWN_BASE

        root_lemma = restored_root if restored_root else (matched_root if matched_root else remainder)
        is_v = _is_verb(root_lemma, tdk)
        
        # apply short root penalty
        if len(remainder) <= _MIN_ROOT_LEN and not is_v and remainder != "":
            root_score -= _SHORT_ROOT_PENALTY
        
        # Aggressive penalty for 1-char roots (e.g. 'd', 'y') to protect 2-char roots
        if len(remainder) == 1 and not is_narrow_root:
            root_score -= 500.0
            

        if root_in_tdk or root_in_domain or restored_root:
            root_token = _root_token(remainder, root_lemma, restored=bool(restored_root))

            p_bonus = _PROGRESSIVE_BONUS if (suffix_label == "-PROG" and is_v) else 0
            v_bonus = _VERB_CHAIN_BONUS if (suffix_label in ("-PST", "-PROG", "-EVID", "-FUT", "-NEG", "-NEG.PROG", "-AOR", "-PASS.AOR") and is_v) else 0
            if is_v and (root_lemma in _PRIMITIVE_VERBS or root_lemma.endswith("mak") or root_lemma.endswith("mek")):
                 v_bonus += 60.0 if root_lemma in {"de", "ye", "et", "git"} else 30.0

            structural_penalty = 0.0
            restoration_bonus = 0.0
            if suffix_label in _FINITE_VERBAL_SUFFIXES and not is_v:
                structural_penalty += _NONVERBAL_FINITE_PENALTY
            if suffix_surface in {"m", "n", "k"} and len(remainder) <= 2 and not is_v:
                structural_penalty += _SHORT_AMBIGUOUS_PERSON_PENALTY
            if restored_kind == "noun" and remainder in _PREFERRED_RESTORED_NOUN_SURFACES and suffix_label == "-ACC":
                restoration_bonus += _PREFERRED_RESTORED_NOUN_BONUS

            # Apply Voice Assimilation Penalty to the whole candidate
            v_alt_penalty = 0
            if suffix_surface and suffix_surface[0] in "tç" and remainder:
                if remainder[-1] not in _UNVOICED_CHARS:
                    v_alt_penalty = 150.0

            candidates.append(SegmentationCandidate(tokens=[root_token, suffix_token], 
                                                     score=root_score + _SUFFIX_BONUS + (len(suffix_surface) * _SUFFIX_LEN_WEIGHT) + p_bonus + v_bonus + restoration_bonus + _TOKEN_COUNT_PENALTY - v_alt_penalty - structural_penalty, 
                                                     source="suffix_chain"))

        if depth < _MAX_SUFFIX_DEPTH - 1:
            sub_candidates = _generate_suffix_candidates(remainder, tdk, domain_roots, depth + 1, allow_rootless, following_label=suffix_label)
            for sc in sub_candidates:
                if sc.score > -200:
                    root_meta = sc.tokens[0].metadata or {}
                    rt = str(root_meta.get("_root_lemma", sc.tokens[0].text))
                    root_is_verb = _is_verb(rt, tdk)
                    pb = _PROGRESSIVE_BONUS if (suffix_label == "-PROG" and root_is_verb) else 0
                    vb = _VERB_CHAIN_BONUS if (suffix_label in ("-PST", "-PROG", "-EVID", "-FUT", "-NEG", "-NEG.PROG", "-AOR", "-PASS.AOR") and root_is_verb) else 0
                    if suffix_label == "-PROG" and any(t.metadata.get("_suffix_label") in ("-NEG", "-NEG.PROG") for t in sc.tokens):
                         vb += 100.0

                    structural_penalty = 0.0
                    if suffix_label in _FINITE_VERBAL_SUFFIXES and not root_is_verb:
                        structural_penalty += _NONVERBAL_FINITE_PENALTY
                    if suffix_surface in {"m", "n", "k"} and len(remainder) <= 2 and not root_is_verb:
                        structural_penalty += _SHORT_AMBIGUOUS_PERSON_PENALTY

                    # Apply Voice Assimilation Penalty here too
                    v_penalty_rec = 0
                    if suffix_surface and suffix_surface[0] in "tç" and remainder:
                        if remainder[-1] not in _UNVOICED_CHARS:
                            v_penalty_rec = 150.0

                    candidates.append(SegmentationCandidate(tokens=sc.tokens + [suffix_token],
                        score=sc.score + _SUFFIX_BONUS + (len(suffix_surface) * _SUFFIX_LEN_WEIGHT) + pb + vb + _TOKEN_COUNT_PENALTY - v_penalty_rec - structural_penalty,
                        source="suffix_chain"))

    return candidates


def generate_candidates(word: str, tdk: set[str], domain_roots: frozenset[str]) -> list[SegmentationCandidate]:
    wl = turkish_lower(word)
    is_caps = is_all_caps_word(word)
    if wl in KNOWN_INTACT:
        return [SegmentationCandidate(tokens=[Token(text=wl, token_type="ROOT", metadata={"_caps": True} if is_caps else {})], score=5000.0, source="known_intact")]
    
    candidates: list[SegmentationCandidate] = []
    
    # Try morphological first (resolves YAPTI)
    suffix_cands = _generate_suffix_candidates(wl, tdk, domain_roots)
    for c in suffix_cands:
        if is_caps and c.tokens:
             c.tokens[0].metadata["_caps"] = True
             if any(t.metadata.get("_suffix_label") == "-PST" for t in c.tokens): c.score += 45.0
        candidates.append(c)

    # Whole word match (exact or ASCII-normalized Turkish variant)
    matched_whole = _resolve_tdk_form(wl, tdk)
    in_tdk = matched_whole is not None
    in_proper = wl in load_proper_nouns()
    if in_tdk or in_proper:
        lemma = matched_whole if matched_whole else wl
        c_score = len(wl)*_ROOT_LEN_WEIGHT + _TDK_BONUS
        if len(wl) <= 5:
            c_score += _WHOLE_WORD_BONUS
        if matched_whole and matched_whole != wl:
            c_score += _ASCII_WHOLE_WORD_BONUS
        candidates.append(SegmentationCandidate(tokens=[_root_token(wl, lemma, is_caps=is_caps)], 
                                                 score=c_score, source="whole_word"))
    else:
        # Check restored whole form
        res_n = restore_noun_stem(wl, tdk); res_v = restore_verb_stem(wl, tdk)
        if res_n or res_v:
             r_text = res_n if res_n else res_v
             candidates.append(SegmentationCandidate(tokens=[Token(text=r_text, token_type="ROOT", metadata={"_restored": True, "_surface_root": wl, "_root_lemma": r_text})],
                                                       score=len(wl)*_ROOT_LEN_WEIGHT + _RESTORED_ROOT_BONUS + _WHOLE_WORD_BONUS + 20.0, source="restored_whole"))
        excess = max(0, len(wl) - _MIN_ROOT_LEN)
        candidates.append(SegmentationCandidate(tokens=[Token(text=wl, token_type="ROOT", metadata={"_caps": True} if is_caps else {})], 
                                                 score=_UNKNOWN_BASE - excess*_UNKNOWN_LEN_PENALTY, source="unknown"))

    if not in_tdk and not has_turkish_chars(wl) and len(wl) >= 3:
        looks_turkishish = wl.isalpha() and not any(ch in wl for ch in "qwx")
        base_foreign_score = _FOREIGN_BASE + len(wl)
        foreign_score = base_foreign_score if (not looks_turkishish and (not candidates or candidates[0].score < 20)) else -200.0
        candidates.append(SegmentationCandidate(tokens=[Token(text=wl, token_type="FOREIGN", metadata={"_foreign": True})], score=foreign_score, source="foreign"))
    
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def select_best_candidate(candidates: list[SegmentationCandidate]) -> SegmentationCandidate:
    if not candidates: return SegmentationCandidate(tokens=[Token(text="", token_type="ROOT")], score=0, source="err")
    best_score = candidates[0].score
    tied = [c for c in candidates if abs(c.score - best_score) < 0.1]
    def _tie_key(c: SegmentationCandidate):
        has_verbal = any(t.metadata.get("_suffix_label") in ("-PROG", "-PST", "-NEG", "-FUT") for t in c.tokens)
        has_restored = any(t.metadata.get("_restored") for t in c.tokens)
        rl = max((len(t.text) for t in c.tokens if t.token_type == "ROOT"), default=0)
        return (len(c.tokens), -rl, -has_verbal, -has_restored)
    tied.sort(key=_tie_key)
    return tied[0]


def _restore_surfaces(best: SegmentationCandidate, original_word: str) -> SegmentationCandidate:
    wl = turkish_lower(original_word)
    # Ensure all tokens correctly slice from original_word
    pos = 0
    for tok in best.tokens:
        t_meta = tok.metadata or {}
        if tok.token_type == "ROOT" and t_meta.get("_restored"):
            # Determine surface length from metadata or current text
            surface_part_len = len(t_meta.get("_surface_root", tok.text))
            original_surface = original_word[pos:pos+surface_part_len]
            if not tok.metadata.get("_root_lemma"):
                tok.metadata["_root_lemma"] = tok.text 
            tok.text = original_surface           
            pos += surface_part_len
        else:
            tlen = len(tok.text)
            tok.text = original_word[pos:pos+tlen]
            pos += tlen
    return best


def segment_word(word: str, tdk: set[str], domain_roots: frozenset[str]) -> list[dict[str, object]]:
    parts = _split_punctuation(word)
    result = []
    for text, ptype in parts:
        if ptype == "PUNCT":
            result.append({"token": text, "token_type": "PUNCT", "morph_pos": 0, "_punct": True})
            continue
        
        # 1. Check Apostrophe FIRST (e-devlet'te)
        if "'" in text or "\u2019" in text:
            apo_tokens = _segment_apostrophe_word(text, tdk, domain_roots)
            result.extend(apo_tokens)
            continue
            
        # 2. Check Hyphenated split SECOND
        if "-" in text and len(text) > 2 and not text.startswith("-") and not text.endswith("-"):
            h_parts = text.split("-")
            for h_idx, h_part in enumerate(h_parts):
                h_cands = generate_candidates(h_part, tdk, domain_roots)
                h_best = _restore_surfaces(select_best_candidate(h_cands), h_part)
                for h_i, h_t in enumerate(h_best.tokens):
                    hd = h_t.to_dict(); hd["morph_pos"] = h_i
                    result.append(hd)
                if h_idx < len(h_parts) - 1:
                    result.append({"token": "-", "token_type": "PUNCT", "morph_pos": 0, "_punct": True})
            continue

        # 3. Default morphological segmentation
        cands = generate_candidates(text, tdk, domain_roots)
        best = _restore_surfaces(select_best_candidate(cands), text)
        for i, t in enumerate(best.tokens):
            d = t.to_dict(); d["morph_pos"] = i
            result.append(d)
    return result


def _segment_apostrophe_word(word: str, tdk: set[str], domain_roots: frozenset[str]) -> list[dict[str, object]]:
    from .apostrophe import is_turkish_base
    apo_pos = max(word.find("'"), word.find("\u2019"))
    base = word[:apo_pos]
    suffix = word[apo_pos + 1:]
    is_tr = is_turkish_base(base)
    is_caps = is_all_caps_word(base)
    tokens = [{"token": base, "token_type": "ROOT" if is_tr else "FOREIGN", "morph_pos": 0, **( {"_caps": True} if is_caps else {}), **( {"_foreign": True} if not is_tr else {})},
              {"token": word[apo_pos], "token_type": "PUNCT", "morph_pos": 0, "_punct": True}]
    if suffix:
        sfx_cands = _generate_suffix_candidates(suffix.lower(), tdk, domain_roots, allow_rootless=True)
        if sfx_cands:
            best_sfx = _restore_surfaces(select_best_candidate(sfx_cands), suffix)
            for i, t in enumerate(best_sfx.tokens):
                d = t.to_dict(); d["morph_pos"] = i + 1; d["_apo_suffix"] = True
                d["token_type"] = "SUFFIX"  # Explicitly force suffix type after apostrophe
                tokens.append(d)
        else:
            tokens.append({"token": suffix, "token_type": "SUFFIX", "morph_pos": 1, "_apo_suffix": True, "_suffix_label": SUFFIX_MAP.get(suffix.lower(), "-SFX")})
    return tokens
