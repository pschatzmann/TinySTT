"""Writes the small deterministic placeholder model + golden outputs used by
the C++ tests until a trained model exists:

  test/data/placeholder_model.tstt  - model container (f32 + int8 weights)
  test/data/model_golden.bin        - T u32 | n_mels u32 | out_frames u32 | classes u32 |
                                      features f32[T*n_mels] | logp_f32 | logp_int8 |
                                      logp_int8_activations
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_container import ContainerWriter  # noqa: E402
import model_numpy as mn  # noqa: E402

PLACEHOLDER = dict(version=1, n_mels=80, conv_channels=32, conv_kernel=5, conv_stride=3,
                   gru_layers=2, hidden=32, n_classes=41)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "test", "data"))
    args = ap.parse_args()
    cfg = PLACEHOLDER
    weights = mn.random_weights(cfg, seed=42)
    w = ContainerWriter()
    mn.add_model(w, cfg, weights, int8=True, float32=True)
    w.add_string("info", "TinySTT placeholder model (random weights, tests only)")
    size = w.write(os.path.join(args.out, "placeholder_model.tstt"))

    rng = np.random.default_rng(7)
    T = 31
    feats = rng.normal(0, 1, (T, cfg["n_mels"])).astype(np.float32)
    lp32 = mn.forward(cfg, weights, feats, use_int8=False)
    lp8 = mn.forward(cfg, weights, feats, use_int8=True)
    lp8a = mn.forward(cfg, weights, feats, use_int8=True, act_quant=True)
    with open(os.path.join(args.out, "model_golden.bin"), "wb") as f:
        f.write(struct.pack("<IIII", T, cfg["n_mels"], lp32.shape[0], lp32.shape[1]))
        f.write(feats.astype("<f4").tobytes())
        f.write(lp32.astype("<f4").tobytes())
        f.write(lp8.astype("<f4").tobytes())
        f.write(lp8a.astype("<f4").tobytes())
    print(f"placeholder model: {size} bytes, golden: {T} frames -> {lp32.shape[0]} outputs")


if __name__ == "__main__":
    main()
