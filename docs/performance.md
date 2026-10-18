# Performance

This page compares the platforms and describes the optimizations they share.
Board settings, platform-specific code, detailed measurements and how to
reproduce them are in the platform documents:

| Platform | Document |
|---|---|
| ESP32-S3, ESP32-P4 | [`platform-esp32.md`](platform-esp32.md) |
| RP2350 | [`platform-rp2350.md`](platform-rp2350.md) |
| Raspberry Pi (Linux) | [`platform-raspberry-pi.md`](platform-raspberry-pi.md) |
| Desktop (Linux, macOS) | [`platform-desktop.md`](platform-desktop.md) |

## Results

Production-shape model (0.9 M parameters, int8; random weights, timing only,
from `research/make_bench_model.py`), 40k-word lexicon, 5 s of audio, one
core. RTF = processing time / audio time; below 1.0 is faster than real time.
The microcontrollers use batches of 4 frames.

| Device | Features | Acoustic model per 30 ms frame | **Stage 1** | Stage 2 (decoder) | int8 kernel |
|---|---:|---:|---:|---:|---|
| ESP32-S3 (240 MHz, octal PSRAM) | 0.087 RTF | 10.1 ms | **0.434 RTF** | 0.014 RTF | esp-nn SIMD |
| ESP32-P4 (400 MHz, rev 3.2) | 0.040 RTF | 3.07 ms | **0.157 RTF** | 0.002 RTF | own PIE kernel |
| RP2350 (150 MHz, 8 MB PSRAM) | 0.133 RTF | 23.5 ms | **0.924 RTF** | 0.004 RTF | Cortex-M33 DSP (SMLAD) |
| Raspberry Pi 4 (1.5 GHz, 32-bit OS) | 0.0027 RTF | 0.39 ms | **0.016 RTF** | 0.0022 RTF | ARM NEON |
| Desktop (i7-4650U) | 0.0008 RTF | 0.58 ms | **0.020 RTF** | 0.0001 RTF | portable (compiler-vectorized) |

All of them run both stages faster than real time on one core. With
`TinySTT`, the decoder runs in its own task (the other core on ESP32).

## Shared optimizations

| Optimization | Where | Effect |
|---|---|---|
| int8 weights **and** int8 activations (`Precision::Auto` = `Int8Activations`) | all | Per-vector symmetric quantization of the activations, int32 dot products: the precondition for SIMD. The numpy reference (`research/model_numpy.py`) implements the same arithmetic. |
| SIMD int8 dot product (`dotS8()`) | per platform | esp-nn (S3), PIE (P4), DSP `SMLAD` (Cortex-M33 / 32-bit ARM), NEON (ARM Linux), else a 4-way unrolled loop compiled with `-O3` even in `-Os` builds. All are bit-identical to the reference (tested). |
| Weight placement (`Placement::Auto`) | microcontrollers | The model is memory-bound once SIMD is used (e.g. ESP32-S3: flash 27 MB/s, PSRAM 59 MB/s, internal RAM 643 MB/s). The GRU recurrent matrices go to internal RAM while `internalRamReserve()` (default 96 KB) stays free; the rest goes to PSRAM. `SpeechToPhonemes::begin()` places the weights last, so the reserve sees the real free memory. |
| Batching (`setBatchFrames(n)`) | microcontrollers | n output frames per pass: the conv, input-projection and output weights are read once per n frames; the recurrence stays sequential. Bit-identical to single frames (`model_batching_is_exact`). Batch 4 adds 90 ms latency. No gain on the desktop, where the caches hold all weights. |

### Features

The FFT is passed in as an object. On the ESP32-S3 and P4, `AudioRealFFT`
measured faster than `AudioEspressifFFT`. Improvements made in audio-tools:

| Improvement | Effect (ESP32-S3) |
|---|---|
| `BufferedWindow(Hann)` instead of `Hann` | `Hann` computes `cos()` for every sample of every frame: 0.62 ms → 0.17 ms per hop |
| Mono framing copies runs into the stride buffer instead of one sample per call | 0.55 → 0.47 ms per hop |
| `toMEL()` computes the spectrum once and builds the filterbank with one allocation | The first call took 40 ms; the steady state is 0.17 ms. `FeatureExtractor::begin()` warms the cache up. |

With these changes the S3 features dropped from 0.156 to 0.087 RTF. The
audio-tools `Hann` window computed 0.54·(1−cos) instead of 0.5·(1−cos); this
was fixed on 2026-10-18 and `research/features.py` follows it.

## Correctness on the devices

`examples/selftest` runs the placeholder model in every precision, placement
and batch size against the numpy goldens, and checks `dotS8()` against the
reference.

| Board | Result |
|---|---|
| ESP32-S3 | all PASS (esp-nn kernel); `TinySTT` with the trained model gives the desktop's transcript (complete pipeline 0.44 RTF) |
| ESP32-P4 | all PASS (PIE kernel); `TinySTT` with the trained model gives the desktop's transcript (complete pipeline 0.16 RTF) |
| RP2350 | all PASS (DSP kernel); `TinySTT` with the trained model gives the desktop's transcript |
| Raspberry Pi 4, desktop | all host test suites pass |

## Accuracy and decoder tuning

Word error rate (WER) = (substituted + deleted + inserted words) / words in the
reference transcript; lower is better. The phoneme error rate (PER) is the same
measure on phonemes and rates the acoustic model alone; "greedy" means the
most likely phoneme per frame, without lexicon or LM.

The shipped model (`data/tinystt.tstt`), on LibriSpeech test-clean:

| Measure | Result |
|---|---|
| Greedy phoneme error rate (500 utterances, int8) | 14.8 % |
| Word error rate, complete C++ pipeline, beam 16 (default) | 36.0 % (2,620 utterances) |
| Word error rate, beam 32 | 32.7 % |
| TinyTTS → TinySTT round trip (`test_roundtrip`, 4 sentences) | 33 % |

The decoder parameters were tuned on the first 100 test-clean utterances
(`research/eval_wer.py`):

| Change | WER (100 utterances) |
|---|---:|
| Initial defaults (beam 16, LM weight 0.8, word bonus 1.5) | 61.9 % |
| Beam 128 instead of 16 | 35.8 % |
| `word_end_beam = 8` (keeps word-boundary hypotheses) at beam 16 | 34.7 % |
| + LM weight 0.5, word bonus 0 (now the defaults) | 31.9 % |

Without a language model the WER is 60–80 %. The LM is essential but weak,
being a bigram from 18,000 training sentences. The Raspberry Pi gives the same
WER as x86 (see [`platform-raspberry-pi.md`](platform-raspberry-pi.md)).

## Design budget

| Stage | Budget | Production shape |
|---|---|---|
| Acoustic model | ≤ 1.5 M MAC/frame | 0.90 M MAC/frame |
| Parameters | ≤ 1.5 M | 0.90 M (≈ 0.95 MB int8) |
| Activations | ≤ 64 KB | 14 KB (batch 4: 36 KB) |
| Queue | ≤ 1 KB | 64 × 16 B |
