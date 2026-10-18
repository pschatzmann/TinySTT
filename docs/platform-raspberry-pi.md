# Raspberry Pi (Linux)

## Boards

| Board | Tested with |
|---|---|
| Raspberry Pi 4 Model B | Cortex-A72, 4 cores at 1.5 GHz, 4 GB, Raspberry Pi OS Bookworm (32-bit userland) |

Any Raspberry Pi with Linux works with the desktop build. A Pi 2/3/4/5 gets
the NEON kernel, a Pi 5 additionally the int8 dot-product instruction.

## Setup

The same CMake build as on the desktop (see [`platform-desktop.md`](platform-desktop.md)):

```bash
# next to each other: TinySTT and arduino-audio-tools (main, 2026-10-18 or later)
cmake -S TinySTT -B build -DTINYSTT_BUILD_TESTS=ON -DTINYSTT_BUILD_DESKTOP=ON
cmake --build build -j4
ctest --test-dir build
```

On a Pi 4 the full build takes about 3 minutes. The `tinystt` command reads a
microphone through miniaudio (ALSA) when built with `-DTINYSTT_DESKTOP_MIC=ON`.

## What is platform specific

| Topic | Raspberry Pi |
|---|---|
| Compiler target | The 32-bit Raspberry Pi OS compiler targets ARMv6 (for the original Pi Zero), without NEON. TinySTT compiles host builds with `-mcpu=native` on ARM (option `TINYSTT_NATIVE`, default ON); `-DTINYSTT_NATIVE=OFF` builds a binary for older Pis |
| int8 dot product | ARM NEON: `vmull_s8` + `vpadalq_s16`, 16 bytes per step; with the dot-product extension (Pi 5) a single `vdotq_s32`. 32- and 64-bit userland |
| Weight placement | none needed (weights stay in RAM) |
| Tasks | `std::thread` |

## Results

Production-shape model (0.9 M parameters, random weights: timing only), 40k
words, 5 s of audio, one core, best of 5 runs.

| | ARMv6 build (DSP kernel) | Native build (NEON) |
|---|---:|---:|
| Features (`AudioRealFFT`, 80 mel) | 0.0030 RTF | 0.0027 RTF |
| Acoustic model, int8, batch 4 | 0.93 ms / 30 ms frame | 0.39 ms |
| **Stage 1** (features + model) | 0.034 RTF | **0.016 RTF** |
| Stage 2 (decoder, 40k words, beam 16) | 0.0023 RTF | 0.0022 RTF |

### Accuracy

The shipped model gives the same word error rate as on x86: 31.9 % on the
first 100 test-clean utterances. 86 of the 100 transcripts are identical
character for character; the others differ in close beam decisions caused by
floating-point details. Decoding these 100 utterances (about 12 minutes of
audio) with one process start each takes 26 s.

### Correctness

All host test suites pass on the Pi, with the same results as on x86.

## Reproducing

```bash
python research/make_bench_model.py --out bench.tstt
./build/desktop/tinystt_benchmark bench.tstt
```
