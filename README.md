# TinySTT

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue?logo=arduino&logoColor=white)](https://www.arduino.cc/reference/en/libraries/)
[![ESP-IDF Component](https://img.shields.io/badge/ESP--IDF-component-blue?logo=espressif&logoColor=white)](idf_component.yml)
[![CMake](https://img.shields.io/badge/CMake-supported-blue?logo=cmake&logoColor=white)](CMakeLists.txt)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

A header-only C++ speech-to-text library for microcontrollers: the counterpart
of [TinyTTS](https://github.com/pschatzmann/TinyTTS). It is split into two
independent stages:

- **Speech → Phonemes** (`SpeechToPhonemes`): log-mel features, then a small streaming CTC phoneme model;
- **Phonemes → Text** (`PhonemesToText`): a lexicon-constrained beam search with a language model or a command grammar.

The stages exchange only phoneme frames through a lock-free queue. They can
run in one loop, on two tasks or cores, or on **two different devices**
connected by a serial line, UDP or ESP-NOW. All audio, FFT and concurrency
functionality comes from
[arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools).
Every model stage is plain hand-written C++, with no TFLite or other
inference runtime. Like TinyTTS, it also builds as a command line program for
your desktop, and it can be tested there with the
[Arduino-Emulator](https://github.com/pschatzmann/Arduino-Emulator).

A trained English model ships in [`data/`](data/README.md): 3.1 MB
with acoustic model, a 25,814-word lexicon and a bigram language model. On
LibriSpeech test-clean the complete pipeline reaches a word error rate of
36 % (beam 16) or 33 % (beam 32). Speech → phonemes runs faster than real
time on the ESP32-S3, ESP32-P4, RP2350, Raspberry Pi 4 and the desktop (see
[Performance](#performance)). See [`docs/limitations.md`](docs/limitations.md)
for what is still missing.


## Usage

`TinySTT` is an audio-tools `AudioOutput`: you write audio into it, and it prints
every recognized utterance as one line to a `Print` such as `Serial`. A sketch
needs four objects and two calls:

1. An audio source, e.g. an `I2SStream` microphone configured for 16 kHz, mono,
   16 bit and started with its `begin()`. Any audio-tools stream works; for other
   formats put a `FormatConverterStream` in front.
2. An FFT object of your choice (e.g. `AudioRealFFT`) and a `ModelContainer`
   with the model data (acoustic model, lexicon, language model in one `.tstt`
   file, loaded from LittleFS, SD or a flash partition).
3. `TinySTT` itself, constructed with the FFT, the model and the output, and a
   `StreamCopy` from the source to `TinySTT`.
4. In `setup()`, `TinySTT::begin()` starts the decoder task (on the other core
   on ESP32). In `loop()`, the `StreamCopy` copies the audio, and speech →
   phonemes runs there.

Optional settings are made through `phonemesToText()` and `speechToPhonemes()`
before `begin()`:
- a command grammar;
- a word list or hotwords;
- the minimum confidence;
- VAD thresholds and batching.

[`examples/stt_i2s_mic`](examples/stt_i2s_mic) is the complete sketch, and
[`examples/stt_commands`](examples/stt_commands) adds a grammar. For everything
in one loop without a task there is `STTPipeline`; for two devices, the stages
are used directly ([`examples/stt_split_sender`](examples/stt_split_sender),
[`examples/stt_split_receiver`](examples/stt_split_receiver)).

On the desktop, the `tinystt` command transcribes WAV files or raw audio from
stdin. It can also run the two stages as two processes connected by a pipe
(see [`docs/platform-desktop.md`](docs/platform-desktop.md)).

## Examples

| Example | What it shows |
|---|---|
| [`stt_i2s_mic`](examples/stt_i2s_mic) | I2S microphone → `TinySTT` → text on `Serial` |
| [`stt_commands`](examples/stt_commands) | Voice commands with a grammar and a minimum confidence |
| [`stt_phonemes`](examples/stt_phonemes) | Step 1 only: the recognized phonemes printed with `PhonemeOutput` (checks microphone and acoustic model) |
| [`stt_split_sender`](examples/stt_split_sender) / [`stt_split_receiver`](examples/stt_split_receiver) | The two stages on two devices, connected by a UART |
| [`timing_benchmark`](examples/timing_benchmark) | Timing of every stage on the device (model in a flash partition) |
| [`selftest`](examples/selftest) | On-device check of all kernels and precisions against the Python reference |

On the desktop, `tinystt` is a command line front end and `tinystt_benchmark`
the timing benchmark (see [`docs/platform-desktop.md`](docs/platform-desktop.md)).

## How it works

```
PCM ─► VAD + FeatureExtractor ─► AcousticModel ─► FrameBuilder ─► [SPSC queue] ─► CtcBeamDecoder ─► text
        (VolumeMeter,             (conv + 2×GRU,   (top-k,              │            (lexicon trie,
         injected FFT, log-mel)    CTC, 30 ms)      blank runs)         │             LM / grammar)
                                           QueueToStream ─► Serial / UDP / ESP-NOW ─► StreamToQueue
```

Stage 1:
- `SpeechToPhonemes` cuts 16 kHz PCM into 10 ms blocks.
- An energy VAD (audio-tools' `VolumeMeter`) finds utterances.
- 80 log-mel bands are computed with the FFT object **you pass in**; `AudioRealFFT` measured fastest on ESP32-S3 and P4.
- A causal conv + GRU model (0.9 M parameters, int8 weights and activations, SIMD on ESP32-S3/P4) outputs CTC phoneme probabilities every 30 ms.
- Only the top-4 phonemes per frame are kept. Near-certain blanks are merged into run-length messages, which cuts the traffic 5–10×.

Stage 2:
- `PhonemesToText` runs a CTC prefix beam search constrained by a pronunciation trie (CMUdict).
- Words are scored with a bigram language model, or with a command grammar such as `turn (on | off) the (light | fan)`.
- It supports runtime word lists, hotword boosting, partial results, N-best results and confidences.

See [`docs/architecture.md`](docs/architecture.md) for the details.

## Performance

Measured on 5 s of audio with the production-size model (0.9 M parameters,
int8) and a 40k-word lexicon. RTF = processing time / audio time; below 1.0
is faster than real time.

| Device | Features | Acoustic model per 30 ms frame | **Stage 1** speech → phonemes | **Stage 2** phonemes → text | Compared with real time | int8 kernel | Weights in |
|---|---:|---:|---:|---:|---:|---|---|
| ESP32-S3 (240 MHz) | 0.087 RTF | 10.1 ms | **0.43 RTF** | 0.014 RTF | 2.2 × faster | esp-nn SIMD | PSRAM + 192 KB internal RAM |
| ESP32-P4 (400 MHz) | 0.040 RTF | 3.1 ms | **0.16 RTF** | 0.002 RTF | 6 × faster | own PIE SIMD kernel | PSRAM + 192 KB internal RAM |
| RP2350 (150 MHz, 8 MB PSRAM) | 0.133 RTF | 23.5 ms | **0.92 RTF** | 0.004 RTF | 1.1 × faster | Cortex-M33 DSP (SMLAD) | PSRAM + 384 KB internal RAM (needs a 48 KB reserve and no OS, see the platform document) |
| Raspberry Pi 4 (1.5 GHz) | 0.003 RTF | 0.39 ms | **0.016 RTF** | 0.002 RTF | 55 × faster | ARM NEON | RAM |
| Desktop (Intel i7-4650U) | 0.001 RTF | 0.58 ms | **0.020 RTF** | 0.0001 RTF | 50 × faster | portable (compiler-vectorized) | RAM |

All values are for one core; the microcontrollers use batches of 4 frames
(+90 ms latency). Stage 1 runs in the caller's task and stage 2 in its own
task, on the other core on ESP32. [`docs/performance.md`](docs/performance.md)
compares the platforms and describes the shared optimizations. Board
settings, detailed measurements and how to reproduce them are in the platform
documents: [`platform-esp32.md`](docs/platform-esp32.md), [`platform-rp2350.md`](docs/platform-rp2350.md),
[`platform-raspberry-pi.md`](docs/platform-raspberry-pi.md) and [`platform-desktop.md`](docs/platform-desktop.md).

**Accuracy** is given as word error rate (WER): the substituted, deleted and
inserted words divided by the number of words in the reference transcript
(lower is better, 0 % is perfect). The phoneme error rate (PER) is the same
measure on phonemes and rates the acoustic model alone. For the shipped model
on LibriSpeech test-clean (2,620 utterances): WER 36.0 % with beam 16 (the default) and 32.7 % with beam 32. The
same transcripts come out on all platforms (the WER was identical on x86 and
the Raspberry Pi). Speech synthesized by TinyTTS is transcribed with a WER of
33 % (2 of 4 test sentences exact).
On the ESP32-S3, ESP32-P4 and RP2350, `TinySTT` with the shipped model
transcribed a 10.4 s LibriSpeech utterance exactly like the desktop, with the
complete pipeline (decoder included) at 0.44, 0.16 and 0.97 RTF.

## Requirements

**Flash (PROGMEM):**

| Part | Size |
|---|---:|
| Program (examples) | 0.1 – 0.2 MB (RP2350), 0.4 – 0.55 MB (ESP32) |
| Acoustic model (int8) | 0.93 MB |
| Lexicon (25,814 words) | 1.42 MB |
| Language model (bigram) | 0.78 MB |
| **Total for free dictation** | **≈ 3.7 MB** |

With a command grammar instead of free dictation, the language model is not
needed. A split sender needs only the acoustic model, and the receiver only
lexicon and LM.

**RAM:**

| Part | Size |
|---|---:|
| Acoustic model state and buffers | 14 KB (36 KB with batches of 4) |
| Features (512-point FFT, 80 mel bands) | ≈ 8 KB |
| Phoneme queue (64 frames) | 1 KB |
| Decoder (beam 16, word history) | up to ≈ 30 KB, plus words added at runtime |
| *Optional, for speed:* GRU recurrent weights in internal RAM | 192 – 384 KB |
| *Optional, for speed:* remaining int8 weights in PSRAM | 0.5 – 0.9 MB |
| Model container | none if used in place from flash (ESP32 partition, embedded array); 3.1 MB PSRAM if loaded from a file system |

Real time on a microcontroller needs about 0.5 MB internal RAM and PSRAM
(ESP32-S3, ESP32-P4, RP2350 with PSRAM). Without PSRAM the weights are read
from flash, which works but is slower. TinySTT needs
[arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools) from
2026-10-18 or later. See [`docs/requirements.md`](docs/requirements.md) for
board settings.

## Documentation

| Document | What's in it |
|---|---|
| [`architecture.md`](docs/architecture.md) | The two stages, the queue, the decoder and the threading options. |
| [`protocol.md`](docs/protocol.md) | `PhonemeFrame` and the wire format that connects two devices. |
| [`phonemes.md`](docs/phonemes.md) | The 41 model classes (blank, silence, 39 ARPAbet phonemes). |
| [`model-data.md`](docs/model-data.md) | The `.tstt` model container: sections, loading, sizes. |
| [`requirements.md`](docs/requirements.md) | Libraries, boards, memory and board settings. |
| [`platform-esp32.md`](docs/platform-esp32.md) | Platform: ESP32-S3 and ESP32-P4 (board settings, SIMD kernels, results). |
| [`platform-rp2350.md`](docs/platform-rp2350.md) | Platform: RP2350 with PSRAM (board settings, DSP kernel, results). |
| [`platform-raspberry-pi.md`](docs/platform-raspberry-pi.md) | Platform: Raspberry Pi with Linux (build, NEON kernel, results). |
| [`platform-desktop.md`](docs/platform-desktop.md) | Platform: desktop (build, tests, the `tinystt` command, benchmark, results). |
| [`platform-esp-idf.md`](docs/platform-esp-idf.md) | Using TinySTT as an ESP-IDF component. |
| [`data/`](data/README.md) | The shipped English model container and its accuracy. |
| [`training.md`](docs/training.md) | Training the acoustic model (Kaggle notebook, Docker), building lexicon and LM. |
| [`research.md`](docs/research.md) | The Python tooling in `research/`. |
| [`performance.md`](docs/performance.md) | Comparison of all platforms, shared optimizations, accuracy and decoder tuning. |
| [`limitations.md`](docs/limitations.md) | Known limitations. |

## Conclusions

- **The split works.** Speech → Phonemes and Phonemes → Text give identical results in one loop, on two tasks over the SPSC queue, and across a fragmented byte stream between two "devices" (tested).
- **Microcontrollers are fast enough.** With int8 weights and activations, the production acoustic model (0.9 M parameters, 30 M MAC/s) plus features runs at 0.43 × real time on an ESP32-S3 and 0.16 × on an ESP32-P4 (measured). On the S3 the bottleneck is memory bandwidth, not arithmetic: placing the weights in PSRAM or internal RAM and batching frames mattered more than SIMD alone.
- **Accuracy is the open front.** The shipped model reaches 36 % WER on LibriSpeech test-clean: usable for commands (grammar mode), not yet for reliable dictation. The biggest gains are expected from a larger language model and from the 38 % of the training data that was skipped because of words missing from the dictionary (see [`docs/training.md`](docs/training.md)).

See [Performance](#performance) for the measured numbers.

## Attribution

The phoneme set and pronunciations come from the
[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict). The
acoustic model is trained on [LibriSpeech](https://www.openslr.org/12)
(CC BY 4.0). Audio, FFT and concurrency are provided by
[arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools).

