"""Builds TinySTT's bigram language model as model container sections.

    python research/build_lm.py --text-from train.tsv --lexicon lexicon.tstt --out lm.tstt

Vocabulary = the lexicon's words (same ids) + <s>, </s>, <unk>. Bigrams with
count < --min-count are pruned; the remaining model is a properly normalized
backoff model with absolute discounting:
    p(w|h) = (c(h,w) - D) / c(h)               for kept bigrams
    p(w|h) = alpha(h) * p_uni(w)               otherwise
    alpha(h) = (1 - sum_kept p(w|h)) / (1 - sum_kept p_uni(w))
Log probabilities are natural logs stored as int16 x 1000 (lm.config scale).

Sections: lm.config U32 [version, vocab (incl. specials), order, scale],
lm.uni_logp I16 [V], lm.uni_backoff I16 [V], lm.bi_offsets U32 [V+1],
lm.bi_words U32 (sorted per history), lm.bi_logp I16.
"""
import argparse
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_container import ContainerWriter, read_container  # noqa: E402
from phonemes import normalize_words  # noqa: E402

VERSION = 1
SCALE = 1000


def lexicon_words(container_path):
    c = read_container(open(container_path, "rb").read())
    off, text = c["lex.word_offsets"], c["lex.word_text"]
    return [text[off[i]:off[i + 1]].decode() for i in range(len(off) - 1)]


def build_sections(words, sentences, min_count=2, discount=0.75, unk_count=1.0):
    ids = {w: i for i, w in enumerate(words)}
    V0 = len(words)
    BOS, EOS, UNK = V0, V0 + 1, V0 + 2
    V = V0 + 3
    uni = collections.Counter()
    bi = collections.Counter()
    for s in sentences:
        seq = [BOS] + [ids.get(w, UNK) for w in s] + [EOS]
        uni.update(seq[1:])
        bi.update(zip(seq[:-1], seq[1:]))
    # unigram with add-one smoothing over the vocabulary (no <s>)
    total = sum(uni.values()) + (V - 1)
    p_uni = np.array([(uni[i] + 1.0) / total for i in range(V)])
    p_uni[BOS] = 0.0
    p_uni /= p_uni.sum()
    hist = collections.Counter()
    for (h, _), c in bi.items():
        hist[h] += c
    kept = collections.defaultdict(dict)
    for (h, w), c in bi.items():
        if c >= min_count:
            kept[h][w] = (c - discount) / hist[h]
    uni_logp = np.full(V, -32.0)
    backoff = np.zeros(V)
    for i in range(V):
        if p_uni[i] > 0:
            uni_logp[i] = math.log(p_uni[i])
    offsets = [0]
    bw, bl = [], []
    for h in range(V):
        succ = kept.get(h, {})
        if succ:
            left = 1.0 - sum(succ.values())
            denom = 1.0 - sum(p_uni[w] for w in succ)
            backoff[h] = math.log(max(left, 1e-9) / max(denom, 1e-9))
            for w in sorted(succ):
                bw.append(w)
                bl.append(math.log(succ[w]))
        offsets.append(len(bw))

    def q(a):
        return np.clip(np.round(np.asarray(a) * SCALE), -32768, 32767).astype("<i2")

    return {
        "lm.config": np.array([VERSION, V, 2, SCALE], dtype="<u4"),
        "lm.uni_logp": q(uni_logp),
        "lm.uni_backoff": q(backoff),
        "lm.bi_offsets": np.array(offsets, dtype="<u4"),
        "lm.bi_words": np.array(bw, dtype="<u4"),
        "lm.bi_logp": q(bl),
    }


def sentences_from(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                text = line.split("\t", 1)[1] if "\t" in line else line
                words = normalize_words(text)
                if words:
                    yield words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-from", nargs="+", required=True, help="manifests or plain text files")
    ap.add_argument("--lexicon", required=True, help="container with lex.* sections")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    words = lexicon_words(args.lexicon)
    sections = build_sections(words, sentences_from(args.text_from), args.min_count)
    w = ContainerWriter()
    for name, value in sections.items():
        w.add(name, value)
    size = w.write(args.out)
    print(f"wrote {args.out}: vocab {len(words) + 3}, {len(sections['lm.bi_words'])} bigrams, {size} bytes")


if __name__ == "__main__":
    main()
