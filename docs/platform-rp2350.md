# RP2350 (Raspberry Pi Pico 2 and compatible boards)

## Boards

| Board | Tested with |
|---|---|
| RP2350B board with PSRAM | "Pico 2 Tiny RP2350B" board: Cortex-M33 at 150 MHz, 8 MB PSRAM (CS = GPIO 19), 16 MB flash, 520 KB RAM |

Without PSRAM the weights are read from flash, which works but is not real
time. The RP2040 (Cortex-M0+, no FPU, no DSP instructions, 264 KB RAM, no
PSRAM) is too slow for this model.

## Setup

Arduino-Pico core, board "Generic RP2350":

| Setting | Value |
|---|---|
| PSRAM | 8 MByte (as fitted) |
| PSRAM CS | **the board's chip-select pin**: GPIO 19 on the tested board; GPIO 47 on Pimoroni Pico Plus 2, PGA2350 and Waveshare RP2350B Plus; GPIO 8 on Adafruit Feather RP2350 |
| Flash | the real flash size, with a file system for `/tinystt.tstt` (e.g. 16 MB with 8 MB FS) |
| Operating System | *None* (recommended): the decoder runs in `TinySTT::write()`; *FreeRTOS SMP*: `TinySTT` runs the decoder in its own task on core 1, but stage 1 is slower (see Results) |

For real time, call
`stt.speechToPhonemes().acousticModel().setInternalRamReserve(48 * 1024)`
before `begin()`. The default reserve (96 KB) leaves room for only one of the two
GRU recurrent matrices in internal RAM, which is safe but slower than real time.

Board labels can be wrong: the tested board was sold as "2 MB flash" but has
16 MB (JEDEC ID `20 40 18`). If PSRAM isn't detected (`rp2040.getPSRAMSize()`
is 0), the CS pin is wrong.

## What is platform specific

| Topic | RP2350 |
|---|---|
| int8 dot product | Cortex-M33 DSP extension: the activations are unpacked once per vector into 16-bit pairs (`SXTB16`); `SMLAD` does two multiply-adds per instruction. Weights in internal RAM are processed two rows per pass; weights from flash/PSRAM one row at a time (flash and PSRAM share the QSPI bus, sequential reads are faster) |
| Weight placement (`Placement::Auto`) | The GRU recurrent matrices (192 KB each) go to internal RAM as long as `internalRamReserve()` stays free, the rest to PSRAM (`pmalloc`). The 520 KB RAM fit both matrices with a 48 KB reserve (21 KB stay free during decoding); the default 96 KB reserve gives one |
| Bottleneck | arithmetic and the shared QSPI bus |
| Tasks | audio-tools provides `Task` only with FreeRTOS; without it `TinySTT` decodes in the caller's task. With FreeRTOS the decoder task runs on core 1 (`loop()` runs on core 0) |
| USB serial | output written before the host opens the port is lost (`examples/selftest` therefore waits for the connection) |

## Results

### Complete pipeline with the trained model

Measured with the previous acoustic model (trained on 100 h); the current
model has the same shape, so time and memory are the same.

`TinySTT` with `data/tinystt.tstt` (lexicon and LM included) on a 10.4 s
LibriSpeech test-clean utterance, 150 MHz. The time covers features, acoustic
model and decoder until the result is printed.

| Setup | GRU weights in internal RAM | Time | RTF | Lowest free heap |
|---|---:|---:|---:|---:|
| No OS, reserve 48 KB | 384 KB | 10.2 s | **0.97** | 21 KB |
| No OS, default reserve (96 KB) | 192 KB | 12.8 s | 1.23 | 218 KB |
| FreeRTOS, reserve 48 KB (decoder task on core 1) | 192 KB | 16.4 s | 1.57 | 182 KB |

The transcript is identical to the desktop's. With FreeRTOS, the 16 KB
decoder stack leaves too little room for the second matrix, and stage 1 is
slower than without an OS even with the same placement (16.4 s vs 12.8 s).
After `end()` and a second `begin()` the decoder keeps about 24 KB of
buffers, so the second matrix no longer fits with a 48 KB reserve.

### Timing per stage

Production-shape model (0.9 M parameters, random weights: timing only), 40k
words, 5 s of audio, 150 MHz (not overclocked), one core.

| | RP2350 |
|---|---:|
| Features (`AudioRealFFT`, 80 mel) | 0.133 RTF |
| Acoustic model, int8, batch 4 | 23.5 ms / 30 ms frame |
| **Stage 1** (features + model) | **0.924 RTF** |
| Stage 2 (decoder, 40k words, beam 16) | 0.004 RTF |

### Optimization steps

| Model weights | ms per 30 ms frame | Model RTF (without features) |
|---|---:|---:|
| In flash | 65.0 | 2.17 |
| In PSRAM | 53.4 | 1.78 |
| Auto (384 KB internal + PSRAM) | 38.7 | 1.29 |
| Auto, batch 4 | 23.5 | 0.78 |
| Auto, batch 8 | 20.7 | 0.69 |

The DSP kernel was tuned with this board: reading two rows per pass is faster
for weights in internal RAM (batch 8: 20.1 vs 23.2 ms/frame), but slower for
weights in flash (114 vs 67 ms/frame), hence the per-matrix choice.

Higher clock settings (176–300 MHz, "Overclock" in the core) exist but were
not measured.

### Correctness

`examples/selftest` passes completely: every precision, placement and batch
size matches the numpy reference (max difference ≤ 1e-6).

A hang found on this board (the board stopped responding on USB) had two
causes, both fixed and verified with the trained model:
- `AcousticModel::begin()` allocated its batch buffers after placing the
  weights, so the internal RAM reserve did not account for them;
- the decoder's word history grew with the length of the utterance (about
  100 KB for 10 s of speech); it is now pruned to what the beam still uses
  (about 30 KB at most).

## Reproducing

`examples/timing_benchmark` loads the container from LittleFS on the RP2350
(`/tinystt.tstt`, e.g. the output of
`python research/make_bench_model.py --out bench.tstt --no-float32`).
Board settings as above.
