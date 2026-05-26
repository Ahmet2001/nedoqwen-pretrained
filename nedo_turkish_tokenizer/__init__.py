"""NedoTurkishTokenizer — self-contained Turkish morphological tokenizer.

Zero external dependencies.  Segments Turkish text into morphologically
meaningful tokens using a candidate-based segmentation engine with a
bundled TDK dictionary, suffix heuristics, and domain-aware vocabulary.

Usage::

    from nedo_turkish_tokenizer import NedoTurkishTokenizer

    tok = NedoTurkishTokenizer()
    tokens = tok.tokenize("İstanbul'da meeting'e katılamadım")
    for t in tokens:
        print(t["token"], t["token_type"], t["morph_pos"])
"""

from .tokenizer import NedoTurkishTokenizer

__all__ = ["NedoTurkishTokenizer"]
__version__ = "2.1.21"
