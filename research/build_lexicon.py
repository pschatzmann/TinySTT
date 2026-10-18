"""Builds TinySTT's static lexicon (phoneme trie) as model container sections.

    python research/build_lexicon.py --vocab-from train.tsv --max-words 40000 --out lexicon.tstt
    python research/build_lexicon.py --words words.txt --out lexicon.tstt

The vocabulary is the most frequent words of the transcripts (or a word list),
restricted to words with a CMUdict pronunciation. Words are sorted bytewise:
the index is the word id shared with the language model. All pronunciation
variants go into the trie; lex.prons keeps the first one per word.
Layout: see src/TinySTT/Lexicon.h.
"""
import argparse
import collections
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_container import ContainerWriter  # noqa: E402
from phonemes import load_cmudict, normalize_words  # noqa: E402


def vocab_from_manifests(paths):
    counts = collections.Counter()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    counts.update(normalize_words(line.split("\t", 1)[1]))
    return counts


def build_sections(words, lexicon):
    """words: iterable of words with pronunciations -> dict of numpy sections."""
    words = sorted(set(words), key=lambda w: w.encode())
    # trie: node = [phone, children{phone: node}, word ids]
    nodes = [[0, {}, []]]
    for wid, w in enumerate(words):
        for pron in lexicon[w]:
            n = 0
            for ph in pron:
                child = nodes[n][1].get(ph)
                if child is None:
                    child = len(nodes)
                    nodes.append([ph, {}, []])
                    nodes[n][1][ph] = child
                n = child
            if wid not in nodes[n][2]:
                nodes[n][2].append(wid)
    # breadth first renumbering so children are contiguous and sorted
    order, index = [0], {0: 0}
    i = 0
    while i < len(order):
        for ph in sorted(nodes[order[i]][1]):
            c = nodes[order[i]][1][ph]
            index[c] = len(order)
            order.append(c)
        i += 1
    node_bytes = bytearray()
    node_words = []
    for old in order:
        ph, children, wids = nodes[old]
        first_child = index[children[min(children)]] if children else 0
        if len(children) > 255 or len(wids) > 255:
            raise ValueError("node fan-out too large")
        node_bytes += struct.pack("<IIBBBB", first_child, len(node_words), ph, len(children), len(wids), 0)
        node_words.extend(wids)
    text = b"".join(w.encode() for w in words)
    word_offsets = np.cumsum([0] + [len(w.encode()) for w in words]).astype("<u4")
    prons = [lexicon[w][0] for w in words]
    pron_offsets = np.cumsum([0] + [len(p) for p in prons]).astype("<u4")
    return words, {
        "lex.nodes": bytes(node_bytes),
        "lex.node_words": np.array(node_words, dtype="<u4"),
        "lex.word_offsets": word_offsets,
        "lex.word_text": text,
        "lex.pron_offsets": pron_offsets,
        "lex.prons": np.array([p for pr in prons for p in pr], dtype=np.uint8),
    }


def add_sections(writer, sections):
    for name, value in sections.items():
        if isinstance(value, bytes):
            writer.add_bytes(name, value)
        else:
            writer.add(name, value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-from", nargs="*", default=[], help="manifests (path<TAB>text)")
    ap.add_argument("--words", help="word list file, one word per line")
    ap.add_argument("--max-words", type=int, default=40000)
    ap.add_argument("--lexicon", help="cmudict.dict (downloaded if missing)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lexicon = load_cmudict(args.lexicon)
    if args.words:
        with open(args.words, encoding="utf-8") as f:
            candidates = [w.strip().lower() for w in f if w.strip()]
    elif args.vocab_from:
        candidates = [w for w, _ in vocab_from_manifests(args.vocab_from).most_common()]
    else:
        candidates = sorted(lexicon)
    vocab = [w for w in candidates if w in lexicon][: args.max_words]
    words, sections = build_sections(vocab, lexicon)
    w = ContainerWriter()
    add_sections(w, sections)
    size = w.write(args.out)
    n_nodes = len(sections["lex.nodes"]) // 12
    print(f"wrote {args.out}: {len(words)} words, {n_nodes} trie nodes, {size} bytes")


if __name__ == "__main__":
    main()
