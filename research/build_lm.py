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
import gzip
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


def count_ngrams(words, sentences, chunk=20_000_000):
    """Unigram counts [V] and the bigram counts as sorted keys h * V + w.

    Counting runs on numpy arrays in chunks, so that corpora with hundreds of
    millions of words (the LibriSpeech LM corpus) fit into memory."""
    ids = {w: i for i, w in enumerate(words)}
    V0 = len(words)
    BOS, EOS, UNK = V0, V0 + 1, V0 + 2
    V = V0 + 3
    uni = np.zeros(V, dtype=np.int64)
    keys = np.zeros(0, dtype=np.int64)
    counts = np.zeros(0, dtype=np.int64)
    buf = []

    def flush():
        nonlocal keys, counts
        if not buf:
            return
        seq = np.concatenate(buf)
        buf.clear()
        # pairs within a sentence: the sentences are separated by -1
        h, w = seq[:-1], seq[1:]
        ok = (h >= 0) & (w >= 0)
        h, w = h[ok], w[ok]
        np.add.at(uni, w, 1)
        k, c = np.unique(h * V + w, return_counts=True)
        k, inv = np.unique(np.concatenate([keys, k]), return_inverse=True)
        merged = np.zeros(len(k), dtype=np.int64)
        np.add.at(merged, inv, np.concatenate([counts, c]))
        keys, counts = k, merged

    pending = 0
    for s in sentences:
        seq = np.fromiter((ids.get(w, UNK) for w in s), dtype=np.int64, count=len(s))
        buf.append(np.concatenate([[BOS], seq, [EOS, -1]]))
        pending += len(s) + 3
        if pending >= chunk:
            flush()
            pending = 0
    flush()
    return V, uni, keys, counts


def build_sections(words, sentences, min_count=2, discount=0.75, unk_count=1.0):
    V, uni, keys, counts = count_ngrams(words, sentences)
    BOS = len(words)
    # unigram with add-one smoothing over the vocabulary (no <s>)
    total = uni.sum() + (V - 1)
    p_uni = (uni + 1.0) / total
    p_uni[BOS] = 0.0
    p_uni /= p_uni.sum()
    h_all = keys // V
    hist = np.bincount(h_all, weights=counts, minlength=V)
    kept = counts >= min_count
    kh, kw = h_all[kept], keys[kept] % V
    kp = (counts[kept] - discount) / hist[kh]
    uni_logp = np.full(V, -32.0)
    uni_logp[p_uni > 0] = np.log(p_uni[p_uni > 0])
    backoff = np.zeros(V)
    left = 1.0 - np.bincount(kh, weights=kp, minlength=V)
    denom = 1.0 - np.bincount(kh, weights=p_uni[kw], minlength=V)
    has = np.bincount(kh, minlength=V) > 0
    backoff[has] = np.log(np.maximum(left[has], 1e-9) / np.maximum(denom[has], 1e-9))
    # keys are sorted: by history, then by word
    offsets = np.concatenate([[0], np.cumsum(np.bincount(kh, minlength=V))])

    def q(a):
        return np.clip(np.round(np.asarray(a) * SCALE), -32768, 32767).astype("<i2")

    return {
        "lm.config": np.array([VERSION, V, 2, SCALE], dtype="<u4"),
        "lm.uni_logp": q(uni_logp),
        "lm.uni_backoff": q(backoff),
        "lm.bi_offsets": offsets.astype("<u4"),
        "lm.bi_words": kw.astype("<u4"),
        "lm.bi_logp": q(np.log(kp)),
    }


def sentences_from(paths, max_lines=0):
    n = 0
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                n += 1
                if max_lines and n > max_lines:
                    return
                text = line.split("\t", 1)[1] if "\t" in line else line
                words = normalize_words(text)
                if words:
                    yield words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-from", nargs="+", required=True, help="manifests or plain text files")
    ap.add_argument("--lexicon", required=True, help="container with lex.* sections")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-lines", type=int, default=0, help="read at most this many lines (0: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    words = lexicon_words(args.lexicon)
    sections = build_sections(words, sentences_from(args.text_from, args.max_lines), args.min_count)
    w = ContainerWriter()
    for name, value in sections.items():
        w.add(name, value)
    size = w.write(args.out)
    print(f"wrote {args.out}: vocab {len(words) + 3}, {len(sections['lm.bi_words'])} bigrams, {size} bytes")


if __name__ == "__main__":
    main()
