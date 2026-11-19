"""What Voikko's Finnish analyser makes of inflected words: ordinary ones, coinages, and English terms inflected in Finnish.

Run from the repository root, on a machine with the system library and its Finnish dictionary installed
(`libvoikko1`, `voikko-fi` on Debian and Ubuntu):

    python investigations/lorebook-lemmas/probe_voikko.py [WORD ...]

For each word, prints whether the spellchecker accepts it and every analysis's base form and word class.
Talks to the C library through `ctypes`, so no Python binding is needed.
"""

import ctypes
import ctypes.util
import sys

maybe_path = ctypes.util.find_library("voikko")
if maybe_path is None:
    sys.exit("libvoikko not found")
lib = ctypes.CDLL(maybe_path)

lib.voikkoInit.restype = ctypes.c_void_p
lib.voikkoInit.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p, ctypes.c_char_p]
lib.voikkoSpellCstr.restype = ctypes.c_int
lib.voikkoSpellCstr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.voikkoAnalyzeWordCstr.restype = ctypes.POINTER(ctypes.c_void_p)
lib.voikkoAnalyzeWordCstr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.voikko_mor_analysis_value_cstr.restype = ctypes.c_char_p
lib.voikko_mor_analysis_value_cstr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.voikko_free_mor_analysis.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

error = ctypes.c_char_p()
handle = lib.voikkoInit(ctypes.byref(error), b"fi", None)
if not handle:
    sys.exit(f"voikkoInit failed: {error.value}")

def analyse(word: str) -> list[tuple[str, str]]:
    """(base form, word class) of each analysis Voikko gives `word`; empty when it has none."""
    analyses = lib.voikkoAnalyzeWordCstr(handle, word.encode("utf-8"))
    results = []
    i = 0
    while analyses[i]:
        def value(key: bytes) -> str:
            raw = lib.voikko_mor_analysis_value_cstr(analyses[i], key)
            return raw.decode("utf-8") if raw else "?"
        results.append((value(b"BASEFORM"), value(b"CLASS")))
        i += 1
    lib.voikko_free_mor_analysis(analyses)
    return results

words = sys.argv[1:] or [
    # ordinary Finnish, including the agglutination test
    "juoksentelisinkohan", "veroja", "verottajalta", "kissoillammekin",
    # coinages that follow Finnish rules but are in no dictionary
    "duolekti", "duolektissa", "duolekteja", "hydravero", "hydraveron", "hydraveroa",
    # compounds of known words, which Finnish forms freely
    "kaksikielisyysongelmassa", "lorekirjastossa",
    # English terms of the glossary, inflected as Finnish speakers inflect loans
    "duolectissa", "duolectin", "co-pterin", "fupataan", "fuppaaminen", "sporkilla", "botwallin",
]
for word in words:
    accepted = bool(lib.voikkoSpellCstr(handle, word.encode("utf-8")))
    print(f"{word:28s} spell={'ok ' if accepted else 'no '} {analyse(word)}")
