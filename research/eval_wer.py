"""Word error rate of the complete C++ pipeline (tinystt CLI) on a manifest.

Every utterance is decoded by the desktop CLI (features, int8 acoustic model,
lexicon + LM beam search - exactly the code that runs on the device), fed
as raw 16 kHz mono s16le on stdin.

    python research/eval_wer.py --cli build/desktop/tinystt --container tinystt.tstt \
        --manifest test-clean.tsv --max-items 200
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phonemes import normalize_words  # noqa: E402


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def load_pcm16k(path):
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if rate != 16000:
        n = int(round(len(x) * 16000 / rate))
        x = np.interp(np.arange(n) * rate / 16000, np.arange(len(x)), x)
    return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="path of the tinystt desktop CLI")
    ap.add_argument("--container", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--args", default="", help="extra CLI options, e.g. '--beam 32'")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--vad", action="store_true", help="keep the voice activity detector on (default: off)")
    args = ap.parse_args()
    errors = total = n = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            path, text = line.rstrip("\n").split("\t", 1)
            ref = normalize_words(text)
            cmd = [args.cli, "--model", args.container] + ([] if args.vad else ["--no-vad"]) + args.args.split() + ["-"]
            out = subprocess.run(cmd, input=load_pcm16k(path), capture_output=True, check=True).stdout
            hyp = normalize_words(out.decode(errors="replace"))
            e = edit_distance(ref, hyp)
            errors += e
            total += len(ref)
            n += 1
            if args.verbose:
                print(f"[{e:2d}/{len(ref):2d}] REF: {' '.join(ref)}\n        HYP: {' '.join(hyp)}", flush=True)
            if n % 25 == 0:
                print(f"{n} utterances, WER {errors / total:.3f}", flush=True)
            if args.max_items and n >= args.max_items:
                break
    print(f"{n} utterances, WER {errors / max(total, 1):.3f} ({errors} errors / {total} words)")


if __name__ == "__main__":
    main()
