# ESP32-S3 and ESP32-P4

## Boards

| Board | Tested with |
|---|---|
| ESP32-S3 | ESP32-S3 (QFN56, rev 0.2), 240 MHz, 8 MB octal PSRAM, 16 MB flash |
| ESP32-P4 | ESP32-P4 rev 3.2, 400 MHz, PSRAM, 32 MB flash |

Both use the Arduino-ESP32 core 3.3; TinySTT can also be built as an ESP-IDF
component (see [`platform-esp-idf.md`](platform-esp-idf.md)).

## Setup

| Setting | ESP32-S3 | ESP32-P4 |
|---|---|---|
| PSRAM | OPI (or QSPI, matching the module) | Enabled |
| Chip variant | – | `ChipVariant=postv3` for rev 3.x chips |
| Partition scheme | ≥ 4 MB LittleFS/FFat for `/tinystt.tstt`, or a data partition for the model (see below) | same |
| USB CDC on boot | Disabled when connected through the UART bridge (e.g. CH343): otherwise `Serial` goes to the native USB port | same |

Boards with an audio codec work through audio-tools' `AudioBoardStream`
(arduino-audio-driver board definitions):
[`examples/stt_waveshare_p4`](../examples/stt_waveshare_p4) uses the
Waveshare ESP32-P4-WIFI6 (ES8311 codec, onboard microphone read as 16 kHz mono) with the model compiled
into the program as a PROGMEM array (create `src/TinySTT/data/tinystt_model_data.h`
once with `python research/export_header.py`),
used in place from flash (Flash Size 32MB, partition scheme with a 13 MB app).
`#define MODEL_SOURCE` selects other sources: `1` maps
`data/tinystt_large.tstt` from the flash partition `tinystt` (partition
scheme "Custom" with the example's `partitions.csv`; write the model once
with `esptool write-flash 0x310000 data/tinystt_large.tstt`), `2` downloads
it over Wi-Fi to the TF card once (`ModelDownloader` with audio-tools'
`URLStream`) and loads it from there into PSRAM. Both decode with beam 64.

The model container can be loaded from a file system into PSRAM, or used in
place from a flash data partition with `ModelPartition` (memory-mapped, no RAM
copy). `examples/timing_benchmark/partitions.csv` defines such a partition
`tinystt` at 0x310000:

```bash
esptool --port /dev/ttyACM0 write-flash 0x310000 tinystt.tstt
```

## What is platform specific

| Topic | ESP32-S3 | ESP32-P4 |
|---|---|---|
| int8 dot product | esp-nn `esp_nn_dot_s8_aligned_esp32s3` (vendored assembly, `src/TinySTT/dsp`) | TinySTT's own PIE (xespv) inline-assembly kernel `dotS8P4()`, using the instruction pattern of Espressif's compiled esp-nn P4 kernel |
| Weight placement (`Placement::Auto`) | PSRAM + one GRU recurrent matrix (192 KB) in internal RAM | same |
| Bottleneck | memory bandwidth | arithmetic until SIMD, then memory bandwidth |
| Tasks | `TinySTT`: decoder task on core 0, `loop()` on core 1 | same |

esp-dsp's int8 dot products are not used: on the S3 it returned wrong results
(found by TinyTTS), and on the P4 it returns a saturated int8 value, which is
useless for 256-element sums.

## Results

### Complete pipeline with the trained model

Measured with the previous acoustic model (trained on 100 h); the current
model has the same shape, so time and memory are the same.

`TinySTT` with `data/tinystt.tstt` (lexicon and LM included) in the `tinystt`
flash partition, on a 10.4 s LibriSpeech test-clean utterance. The time covers
features, acoustic model and decoder until the result is printed.

| | ESP32-S3 | ESP32-P4 |
|---|---:|---:|
| Time | 4.58 s (**0.44 RTF**) | 1.63 s (**0.16 RTF**) |
| Weights | 192 KB internal RAM, 686 KB PSRAM | same |
| Lowest free internal RAM | 95 KB | 184 KB |
| Dropped queue frames | 0 | 0 |

Both give the desktop's transcript (and the same phoneme sequence).

### Timing per stage

Production-shape model (0.9 M parameters, random weights: timing only), 40k
words, 5 s of audio, one core. RTF = processing time / audio time.

| | ESP32-S3 | ESP32-P4 |
|---|---:|---:|
| Features (`AudioRealFFT`, 80 mel) | 0.087 RTF | 0.040 RTF |
| Acoustic model, int8, batch 4 | 10.1 ms / 30 ms frame | 3.07 ms |
| **Stage 1** (features + model) | **0.434 RTF** | **0.157 RTF** |
| Stage 2 (decoder, 40k words, beam 16) | 0.014 RTF | 0.002 RTF |

### ESP32-S3: optimization steps

| Step | ms/frame | RTF |
|---|---:|---:|
| float32 weights in flash | 141.6 | 4.73 |
| int8 weights × float activations, flash | 79.1 | 2.64 |
| + int8 activations, esp-nn SIMD | 37.7 | 1.26 |
| + weights copied to PSRAM | 19.6 | 0.65 |
| + one GRU recurrent matrix (192 KB) in internal RAM | 16.5 | 0.55 |
| + batches of 4 output frames | 10.3 | 0.35 |

Once SIMD is used, the S3 is **memory-bandwidth bound**. An int8 768×256
matrix-vector product:

| Weights in | int8 × float (scalar) | int8 SIMD | Bandwidth |
|---|---:|---:|---:|
| Flash (mmap) | 9.6 ms | 7.3 ms | 27 MB/s |
| PSRAM | 8.7 ms | 3.3 ms | 59 MB/s |
| Internal RAM | 6.6 ms | 0.31 ms | 643 MB/s |

### ESP32-P4: optimization steps

| Kernel (768×256 int8) | Time |
|---|---:|
| Plain loop (`-Os`) | 5.03 ms |
| Plain loop, `-O3` | 4.05 ms |
| 4 accumulators, `-O3` (portable fallback) | 2.48 ms |
| float32 esp-dsp | 5.61 ms |
| **PIE SIMD `dotS8P4`** | **0.26 ms** (749 MB/s) |

| Step | Model, batch 4 | Stage 1 RTF |
|---|---:|---:|
| Portable int8 loop, `-Os` (Arduino default) | 25.9 ms/frame | 0.91 |
| 4-way unrolled loop + `Ops.h` with `-O3` | 13.1 ms/frame | 0.49 |
| PIE SIMD kernel | 3.07 ms/frame | 0.157 |

With SIMD the P4 also becomes memory-bound for weights in flash: 28.6 ms per
frame from flash, 6.3 ms from PSRAM, 5.6 ms with Auto placement, 3.1 ms with
batches of 4.

### Correctness

`examples/selftest` passes completely on both chips: every precision,
placement and batch size matches the numpy reference (max difference ≤ 1e-6).
On the P4, the PIE dot product also matched the reference on 2,000 random
vectors and the extreme values.

## Reproducing

```bash
python research/make_bench_model.py --out bench.tstt   # with float32 weights for the float32 rows
# S3: esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=custom
# P4: esp32:esp32:esp32p4:ChipVariant=postv3,PSRAM=enabled,FlashSize=32M,PartitionScheme=custom
arduino-cli compile -b <fqbn> -u -p /dev/ttyACM0 examples/timing_benchmark
esptool --port /dev/ttyACM0 write-flash 0x310000 bench.tstt
```

`examples/selftest` checks the kernels and precisions against the Python
reference.
