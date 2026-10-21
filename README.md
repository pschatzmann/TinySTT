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
[arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools)
(2026-10-18 or later). Every model stage is plain hand-written C++, with no TFLite or other
inference runtime. Like TinyTTS, it also builds as a command line program for
your desktop, and it can be tested there with the
[Arduino-Emulator](https://github.com/pschatzmann/Arduino-Emulator).

Trained English models ship in [`data/`](data/README.md) (see
[Models](#models)). On LibriSpeech test-clean the complete pipeline reaches a
word error rate of 30 % with the 3.1 MB default model, or 26 % with the
9.5 MB large model (see [Accuracy](#accuracy)). Speech → phonemes runs faster than real
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
   with one of the [models](#models) (acoustic model, lexicon and language
   model in one `.tstt` file, loaded from LittleFS or SD, used in place from a
   flash partition or a PROGMEM array, or downloaded with `ModelDownloader`).
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

## Models

The models in [`data/`](data/README.md) share the same acoustic model (0.9 M
parameters, int8, trained on 460 h of LibriSpeech) and differ in vocabulary
and language model:

| File | Size | Lexicon | Language model | Beam | WER (LibriSpeech test-clean) |
|---|---:|---|---|---:|---:|
| `data/tinystt.tstt` | 3.1 MB | 25,814 words (CMUdict) | bigram from the training transcripts | 32 (default) | 29.6 % |
| `data/tinystt_large.tstt` | 9.5 MB | 50,000 words (LibriSpeech lexicon) | bigram from the LibriSpeech LM corpus (800 M words) | 64 | **25.7 %** |

The default model fits any board with 4 MB flash or PSRAM; the large one
needs about 10 MB and runs on the ESP32-S3 and ESP32-P4 (flash and RAM
figures: [`docs/requirements.md`](docs/requirements.md)). A model is loaded
from a file system, used in place from flash, compiled into the program
([`docs/model-headers.md`](docs/model-headers.md)) or downloaded over Wi-Fi
(`ModelDownloader`); see [`docs/model-data.md`](docs/model-data.md). Your own
vocabulary, language model or grammar: see [`docs/training.md`](docs/training.md).

## Examples

| Example | What it shows |
|---|---|
| [`stt_i2s_mic`](examples/stt_i2s_mic) | I2S microphone → `TinySTT` → text on `Serial` |
| [`stt_waveshare_p4`](examples/stt_waveshare_p4) | Waveshare ESP32-P4-WIFI6: onboard microphone (ES8311) via `AudioBoardStream`; the model is downloaded over Wi-Fi to the TF card once (`ModelDownloader`) and loaded from there |
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

## Accuracy

Measured on LibriSpeech test-clean (2,620 utterances, read English speech)
with the complete C++ pipeline:

| Model | Beam | Word error rate | Phoneme error rate (acoustic model, greedy) |
|---|---:|---:|---:|
| `data/tinystt.tstt` (3.1 MB) | 32 (default) | 29.6 % | 15.0 % |
| `data/tinystt_large.tstt` (9.5 MB) | 64 | **25.7 %** | 15.0 % |

The word error rate (WER) is the number of substituted, deleted and inserted
words divided by the number of words in the reference transcript (lower is
better, 0 % is perfect). The phoneme error rate (PER) is the same measure on
phonemes and rates the acoustic model alone; both models share it.

- **The same on every platform:** the WER was identical on x86 and the Raspberry Pi. On the ESP32-S3, ESP32-P4 and RP2350, `TinySTT` with `data/tinystt.tstt` transcribed a 10.4 s LibriSpeech utterance exactly like the desktop, with the complete pipeline (decoder included) at 0.44, 0.16 and 0.97 RTF.
- **Synthesized speech:** speech from TinyTTS is transcribed with a WER of 20 % (3 of 4 test sentences exact).
- **Progress:** the first model (100 h of training data, small lexicon and LM, beam 16) had 36 %. Beam 32, 460 h of training data and the large lexicon and LM brought it to 26 %.
- **Commands:** with a grammar instead of the language model, only the words of the grammar can be recognized, which is far more reliable for voice control (see [`examples/stt_commands`](examples/stt_commands)).

[`docs/performance.md`](docs/performance.md#accuracy-and-decoder-tuning) has
the decoder tuning and the language model experiments,
[`docs/training.md`](docs/training.md) the training results.

## Documentation

| Document | What's in it |
|---|---|
| [`architecture.md`](docs/architecture.md) | The two stages, the queue, the decoder and the threading options. |
| [`protocol.md`](docs/protocol.md) | `PhonemeFrame` and the wire format that connects two devices. |
| [`phonemes.md`](docs/phonemes.md) | The 41 model classes (blank, silence, 39 ARPAbet phonemes). |
| [`model-data.md`](docs/model-data.md) | The `.tstt` model container: sections, loading, sizes. |
| [`model-headers.md`](docs/model-headers.md) | Compiling a model into the program: creating the PROGMEM headers with `research/export_header.py`. |
| [`requirements.md`](docs/requirements.md) | Libraries, boards, memory and board settings. |
| [`platform-esp32.md`](docs/platform-esp32.md) | Platform: ESP32-S3 and ESP32-P4 (board settings, SIMD kernels, results). |
| [`platform-rp2350.md`](docs/platform-rp2350.md) | Platform: RP2350 with PSRAM (board settings, DSP kernel, results). |
| [`platform-raspberry-pi.md`](docs/platform-raspberry-pi.md) | Platform: Raspberry Pi with Linux (build, NEON kernel, results). |
| [`platform-desktop.md`](docs/platform-desktop.md) | Platform: desktop (build, tests, the `tinystt` command, benchmark, results). |
| [`platform-esp-idf.md`](docs/platform-esp-idf.md) | Using TinySTT as an ESP-IDF component. |
| [`data/`](data/README.md) | The shipped English models (default and large) and their accuracy. |
| [`training.md`](docs/training.md) | Training the acoustic model (Kaggle notebook, Docker), building lexicon and LM. |
| [`research.md`](docs/research.md) | The Python tooling in `research/`. |
| [`performance.md`](docs/performance.md) | Comparison of all platforms, shared optimizations, accuracy and decoder tuning. |
| [`limitations.md`](docs/limitations.md) | Known limitations. |

## Conclusions

- **The split works.** Speech → Phonemes and Phonemes → Text give identical results in one loop, on two tasks over the SPSC queue, and across a fragmented byte stream between two "devices" (tested).
- **Microcontrollers are fast enough.** With int8 weights and activations, the production acoustic model (0.9 M parameters, 30 M MAC/s) plus features runs at 0.43 × real time on an ESP32-S3 and 0.16 × on an ESP32-P4 (measured). On the S3 the bottleneck is memory bandwidth, not arithmetic: placing the weights in PSRAM or internal RAM and batching frames mattered more than SIMD alone.
- **Accuracy is the open front.** The default model reaches 30 % WER on LibriSpeech test-clean, the large model 26 %: usable for commands (grammar mode), not yet for reliable dictation. Training on 460 h instead of 100 h and the larger lexicon and language model brought it from 33 % to 26 %; the dev error rate was still falling when the training stopped, so more epochs are the next step (see [`docs/training.md`](docs/training.md)).

See [Performance](#performance) for the measured numbers.

## Attribution

The phoneme set and pronunciations come from the
[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict). The
acoustic model is trained on [LibriSpeech](https://www.openslr.org/12)
(CC BY 4.0). Audio, FFT and concurrency are provided by
[arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools).

