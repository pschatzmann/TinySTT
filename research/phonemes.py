"""Phoneme inventory and CMU dictionary helpers (mirror of src/TinySTT/Phonemes.h)."""
import os
import re
import urllib.request

PHONEMES = ["<b>", "sil", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
            "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
            "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
BLANK = 0
SILENCE = 1
ID = {p: i for i, p in enumerate(PHONEMES)}

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"


def phone_id(symbol):
    """ARPAbet symbol (stress digit ignored) -> id, None if unknown."""
    s = symbol.upper().rstrip("012")
    return ID.get(s)


def load_cmudict(path=None, download_dir=None):
    """Returns {word: [[ids...], ...]} (all pronunciation variants, lower case words)."""
    if path is None:
        download_dir = download_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(download_dir, exist_ok=True)
        path = os.path.join(download_dir, "cmudict.dict")
        if not os.path.exists(path):
            print(f"downloading {CMUDICT_URL}")
            urllib.request.urlretrieve(CMUDICT_URL, path)
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
            ids = [phone_id(p) for p in parts[1:]]
            if any(i is None for i in ids) or not ids:
                continue
            lex.setdefault(word, []).append(ids)
    return lex


def normalize_words(text):
    """Transcript -> list of lower case words (letters and apostrophes)."""
    return re.findall(r"[a-z']+", text.lower())


def text_to_phonemes(text, lexicon):
    """First pronunciation of every word; None if a word is not in the lexicon."""
    out = []
    for w in normalize_words(text):
        prons = lexicon.get(w)
        if not prons:
            return None
        out.extend(prons[0])
    return out
