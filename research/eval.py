"""Phoneme error rate (greedy CTC) of an exported container on a manifest,
using the numpy reference of the device model (int8 weights if present).

    python research/eval.py --container tinystt.tstt --manifest test-clean.tsv --max-items 500
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat  # noqa: E402
import model_numpy as mn  # noqa: E402
from model_container import read_container  # noqa: E402
from phonemes import load_cmudict, text_to_phonemes  # noqa: E402


def weights_from_container(c, int8):
    cfg = mn.config_from_array(c["model.config"])
    w = {}
    for wname, bname in mn.dense_names(cfg):
        if wname in c and not int8:
            w[wname] = c[wname]
        else:  # dequantize; numpy forward re-quantizes identically
            q, s = c[wname + ".q"], c[wname + ".scale"]
            w[wname] = q.reshape(len(s), -1).astype(np.float32) * s[:, None]
        w[bname] = c[bname]
    return cfg, w


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--float32", action="store_true", help="use float32 weights if stored")
    ap.add_argument("--lexicon", help="cmudict.dict (downloaded if missing)")
    args = ap.parse_args()
    import train  # audio loading
    c = read_container(open(args.container, "rb").read())
    has_q = "conv.weight.q" in c
    cfg, w = weights_from_container(c, int8=has_q and not args.float32)
    mean, inv_std = c.get("features.mean"), c.get("features.inv_std")
    lexicon = load_cmudict(args.lexicon)
    errors = total = n = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            path, text = line.rstrip("\n").split("\t", 1)
            ref = text_to_phonemes(text, lexicon)
            if not ref:
                continue
            x = feat.log_mel(train.load_audio(path), mean=mean, inv_std=inv_std)
            logp = mn.forward(cfg, w, x, use_int8=has_q and not args.float32)
            hyp, last = [], None
            for i in logp.argmax(-1):
                if i != last and i != 0:
                    hyp.append(int(i))
                last = i
            errors += edit_distance(hyp, ref)
            total += len(ref)
            n += 1
            if args.max_items and n >= args.max_items:
                break
    print(f"{n} utterances, PER {errors / max(total, 1):.3f} ({'int8' if has_q and not args.float32 else 'float32'})")


if __name__ == "__main__":
    main()
