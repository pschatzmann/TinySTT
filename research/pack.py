"""Merges model containers (acoustic model, lexicon, LM, grammars) into one.

    python research/pack.py model.tstt lexicon.tstt lm.tstt --out tinystt.tstt
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_container import ContainerWriter, read_container  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w = ContainerWriter()
    seen = set()
    for path in args.inputs:
        for name, value in read_container(open(path, "rb").read()).items():
            if name in seen:
                if name == "info":
                    continue
                raise SystemExit(f"duplicate section {name} in {path}")
            seen.add(name)
            if isinstance(value, bytes):
                w.add_bytes(name, value)
            else:
                w.add(name, value)
    size = w.write(args.out)
    print(f"wrote {args.out}: {len(seen)} sections, {size} bytes")


if __name__ == "__main__":
    main()
