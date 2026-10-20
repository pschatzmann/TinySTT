"""Converts the unigrams and bigrams of an ARPA language model into TinySTT's
bigram language model sections (same layout as build_lm.py).

    python research/arpa_to_lm.py --arpa 3-gram.pruned.3e-7.arpa.gz --lexicon lexicon.tstt --out lm.tstt

This makes large public models usable, e.g. the pruned LibriSpeech models
(www.openslr.org/11, trained on 800 M words). The vocabulary is the
lexicon's words (same ids) + <s>, </s>, <unk>; bigrams with a word outside
the vocabulary are dropped. Higher orders are ignored. ARPA backoff semantics
match the device: p(w|h) = bigram if present, else backoff(h) * p(w).
--min-logp drops bigrams with a lower log10 probability (smaller model).
"""
import argparse
import gzip
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lm import SCALE, VERSION, lexicon_words  # noqa: E402
from model_container import ContainerWriter  # noqa: E402

LN10 = math.log(10.0)


def read_arpa(path, ids, min_logp=-99.0):
    """Returns (unigrams {id: (logp, backoff)}, bigrams [(h, w, logp)]) in log10."""
    opener = gzip.open if path.endswith(".gz") else open
    uni, bi = {}, []
    order = 0
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("\\"):
                order = int(line[1]) if line[1].isdigit() else 0
                if order > 2:
                    break
                continue
            if order == 0:
                continue
            parts = line.split()
            logp = float(parts[0])
            words = parts[1:1 + order]
            backoff = float(parts[1 + order]) if len(parts) > 1 + order else 0.0
            keys = [ids.get(w.lower()) for w in words]
            if any(k is None for k in keys):
                continue
            if order == 1:
                uni[keys[0]] = (logp, backoff)
            elif logp >= min_logp:
                bi.append((keys[0], keys[1], logp))
    return uni, bi


def build_sections(words, arpa, min_logp=-99.0):
    V0 = len(words)
    BOS, EOS, UNK = V0, V0 + 1, V0 + 2
    V = V0 + 3
    ids = {w: i for i, w in enumerate(words)}
    ids.update({"<s>": BOS, "</s>": EOS, "<unk>": UNK})
    uni, bi = read_arpa(arpa, ids, min_logp)
    unk_logp = uni.get(UNK, (-7.0, 0.0))[0]
    uni_logp = np.full(V, unk_logp * LN10)
    backoff = np.zeros(V)
    for i, (lp, bo) in uni.items():
        uni_logp[i] = max(lp, -32.0 / LN10) * LN10
        backoff[i] = bo * LN10
    uni_logp[BOS] = -32.0  # never predicted
    bi.sort()
    h = np.array([b[0] for b in bi], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(h, minlength=V))])

    def q(a):
        return np.clip(np.round(np.asarray(a) * SCALE), -32768, 32767).astype("<i2")

    return {
        "lm.config": np.array([VERSION, V, 2, SCALE], dtype="<u4"),
        "lm.uni_logp": q(uni_logp),
        "lm.uni_backoff": q(backoff),
        "lm.bi_offsets": offsets.astype("<u4"),
        "lm.bi_words": np.array([b[1] for b in bi], dtype="<u4"),
        "lm.bi_logp": q(np.array([b[2] for b in bi]) * LN10),
    }, len(uni)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arpa", required=True, help="ARPA model (.arpa or .arpa.gz)")
    ap.add_argument("--lexicon", required=True, help="container with lex.* sections")
    ap.add_argument("--min-logp", type=float, default=-99.0, help="drop bigrams below this log10 probability")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    words = lexicon_words(args.lexicon)
    sections, n_uni = build_sections(words, args.arpa, args.min_logp)
    w = ContainerWriter()
    for name, value in sections.items():
        w.add(name, value)
    size = w.write(args.out)
    print(f"wrote {args.out}: vocab {len(words) + 3} ({n_uni} in the ARPA model), "
          f"{len(sections['lm.bi_words'])} bigrams, {size} bytes")


if __name__ == "__main__":
    main()
