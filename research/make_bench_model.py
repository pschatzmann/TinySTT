"""Production-shape container for on-device benchmarks (random weights:
timing only, no recognition). Includes float32 + int8 weights, feature
normalization and a CMUdict lexicon of --words words.

    python research/make_bench_model.py --out bench.tstt
    # flash into the 'tinystt' partition of examples/timing_benchmark/partitions.csv:
    esptool --port /dev/ttyACM0 write-flash 0x310000 bench.tstt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lexicon  # noqa: E402
import model_numpy as mn  # noqa: E402
from model_container import ContainerWriter  # noqa: E402
from phonemes import load_cmudict  # noqa: E402

PRODUCTION = dict(version=1, n_mels=80, conv_channels=256, conv_kernel=5, conv_stride=3,
                  gru_layers=2, hidden=256, n_classes=41)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=40000)
    ap.add_argument("--no-float32", action="store_true")
    args = ap.parse_args()
    weights = mn.random_weights(PRODUCTION, seed=1)
    w = ContainerWriter()
    mn.add_model(w, PRODUCTION, weights, int8=True, float32=not args.no_float32,
                 mean=np.zeros(80, np.float32), inv_std=np.full(80, 0.2, np.float32))
    lex = load_cmudict()
    words = sorted(lex)[:: max(1, len(lex) // args.words)][: args.words]
    _, sections = build_lexicon.build_sections(words, lex)
    build_lexicon.add_sections(w, sections)
    w.add_string("info", "TinySTT benchmark container (random weights)")
    size = w.write(args.out)
    print(f"wrote {args.out}: {size} bytes ({len(words)} words)")


if __name__ == "__main__":
    main()
