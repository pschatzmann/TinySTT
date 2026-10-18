"""Bit-faithful numpy re-implementation of TinySTT's FeatureExtractor.

The device computes features with the arduino-audio-tools FFT API
(AudioFFTBase with stride, Hann window, toMEL with the cached fractional
filterbank). Training must see exactly the same features, so this module
reproduces every step, including audio-tools' quirks:

* the Hann window 0.5 * (1 - cos(2 pi i / (N-1))) is evaluated in float32 and
  mirrored around the center (see AudioTools/FFT/FFTWindows.h);
* the window is applied to the int16 sample and the product is truncated back
  to int16 (AudioFFTBase::windowedSample), then scaled by 1/32767;
* only bins 0 .. N/2-1 are used (the Nyquist bin is dropped);
* mel filters use fractional HTK-style triangle weights, and a filter narrower
  than one bin falls back to the bin nearest to its center.

Usage:
    python research/features.py --golden test/data/features_golden.bin
"""
import argparse
import struct

import numpy as np

SAMPLE_RATE = 16000
FFT_LENGTH = 512
HOP = 160
MEL_BINS = 80
MIN_FREQ = 0.0
MAX_FREQ = 8000.0
LOG_EPS = 1e-6


def audio_tools_hann(n=FFT_LENGTH):
    """audio_tools::Hann::factor() for idx 0..n-1 (float32 like the C++).
    The window is cached by BufferedWindow (computed once with the same formula)."""
    samples_minus_1 = np.float32(n - 1)
    half = n // 2

    def internal(idx):
        ratio = np.float32(idx) / samples_minus_1
        return np.float32(0.5) * (np.float32(1.0) - np.cos(np.float32(6.28318531) * ratio, dtype=np.float32))

    w = np.empty(n, dtype=np.float32)
    for i in range(n):
        v = internal(i) if i <= half else internal(n - i - 1)
        w[i] = min(v, np.float32(1.0))
    return w


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels=MEL_BINS, n_fft=FFT_LENGTH, sr=SAMPLE_RATE, fmin=MIN_FREQ, fmax=MAX_FREQ):
    """Dense [n_mels, n_fft/2] matrix equal to AudioFFTBase::updateMelFilterbank()."""
    bins = n_fft // 2
    bin_hz = np.float32(sr) / np.float32(n_fft)
    min_mel = np.float32(hz_to_mel(np.float32(fmin)))
    step = (np.float32(hz_to_mel(np.float32(fmax))) - min_mel) / np.float32(n_mels + 1)
    fb = np.zeros((n_mels, bins), dtype=np.float32)
    for i in range(n_mels):
        left = np.float32(mel_to_hz(min_mel + np.float32(i) * step))
        center = np.float32(mel_to_hz(min_mel + np.float32(i + 1) * step))
        right = np.float32(mel_to_hz(min_mel + np.float32(i + 2) * step))
        any_bin = False
        for j in range(bins):
            f = np.float32(j) * bin_hz
            if f <= left or f >= right:
                continue
            w = (f - left) / (center - left) if f <= center else (right - f) / (right - center)
            fb[i, j] = w
            any_bin = True
        if not any_bin:
            j = min(int(center / bin_hz + 0.5), bins - 1)
            fb[i, j] = 1.0
    return fb


_WINDOW = None
_FILTERBANK = None


def log_mel(samples, power=True, mean=None, inv_std=None):
    """int16 mono 16 kHz samples -> [frames, MEL_BINS] float32 log-mel."""
    global _WINDOW, _FILTERBANK
    if _WINDOW is None:
        _WINDOW = audio_tools_hann()
        _FILTERBANK = mel_filterbank()
    samples = np.asarray(samples, dtype=np.int16)
    if len(samples) < FFT_LENGTH:
        return np.zeros((0, MEL_BINS), dtype=np.float32)
    n_frames = 1 + (len(samples) - FFT_LENGTH) // HOP
    idx = np.arange(FFT_LENGTH)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = samples[idx].astype(np.float32)
    windowed = np.trunc(frames * _WINDOW[None, :]).astype(np.int16).astype(np.float32)
    x = windowed * np.float32(1.0 / 32767.0)
    spec = np.fft.rfft(x.astype(np.float64), axis=1)[:, : FFT_LENGTH // 2]
    mag2 = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mag = mag2 if power else np.sqrt(mag2)
    mel = mag @ _FILTERBANK.T
    feats = np.log(mel + np.float32(LOG_EPS)).astype(np.float32)
    if mean is not None:
        feats = (feats - mean[None, :]) * inv_std[None, :]
    return feats


def test_signal(n=16000, seed=1234):
    """Deterministic chirp + LCG noise; identical generator in test_features.cpp."""
    out = np.empty(n, dtype=np.int16)
    state = seed & 0xFFFFFFFF
    for i in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        noise = ((state >> 16) & 0x7FFF) - 16384  # [-16384, 16383]
        t = i / SAMPLE_RATE
        chirp = np.sin(2 * np.pi * (200.0 * t + 1900.0 * t * t))
        out[i] = int(np.clip(round(12000.0 * chirp + 0.1 * noise), -32768, 32767))
    return out


def write_golden(path):
    """samples count u32 | frames u32 | bins u32 | int16 samples | float32 features."""
    samples = test_signal()
    feats = log_mel(samples)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", len(samples), feats.shape[0], feats.shape[1]))
        f.write(samples.astype("<i2").tobytes())
        f.write(feats.astype("<f4").tobytes())
    print(f"wrote {path}: {len(samples)} samples, {feats.shape[0]} frames x {feats.shape[1]} bins")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", help="write the golden feature file used by test_features")
    args = ap.parse_args()
    if args.golden:
        write_golden(args.golden)
    else:
        ap.print_help()
