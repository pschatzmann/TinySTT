# Desktop (Linux, macOS)

The desktop build has the host tests, the `tinystt` command and the
`tinystt_benchmark` timing benchmark. The Raspberry Pi uses the same build
(see [`platform-raspberry-pi.md`](platform-raspberry-pi.md)).

## System

| System | Tested with |
|---|---|
| x86-64 Linux | Intel i7-4650U (1.7 GHz, 2014 laptop), Ubuntu 24.04, g++ 13, CMake 3.28 |

## Setup

```bash
cmake -S . -B build -DTINYSTT_BUILD_TESTS=ON -DTINYSTT_BUILD_DESKTOP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- arduino-audio-tools is taken from the sibling folder `../arduino-audio-tools` if present, otherwise fetched.
- The tests run against the [Arduino-Emulator](https://github.com/pschatzmann/Arduino-Emulator) (`TINYSTT_USE_ARDUINO_EMULATOR=ON`, default), fetched by audio-tools. To use a local checkout, pass `-DFETCHCONTENT_SOURCE_DIR_ARDUINO_EMULATOR=/path/to/Arduino-Emulator`.
- The CLI and the benchmark use audio-tools' lightweight emulation, because they need `argc`/`argv`.

| Option | Effect |
|---|---|
| `TINYSTT_SANITIZE=address\|thread` | Sanitizer build of the tests. TSan on recent kernels needs `setarch $(uname -m) -R ./build/test/test_queue`. |
| `TINYSTT_ROUNDTRIP=ON` | TinyTTS → TinySTT round-trip test (needs `../TinyTTS`). Runs when `TINYSTT_MODEL` points to a trained container, e.g. `TINYSTT_MODEL=data/tinystt.tstt`: WER 33 % on 4 synthesized sentences. |
| `TINYSTT_DESKTOP_MIC=ON` | Live microphone in the CLI (fetches miniaudio). |
| `TINYSTT_NATIVE=ON` (default) | On ARM Linux: compile for the build machine's CPU (`-mcpu=native`, enables NEON on a 32-bit Raspberry Pi OS). |

`TINYSTT_TEST_FILTER=<substring>` runs only matching tests of a test binary.

### The `tinystt` command

```bash
tinystt --model data/tinystt.tstt speech.wav             # any WAV, converted to 16 kHz mono
tinystt --model data/tinystt.tstt --nbest 3 speech.wav   # N-best with confidences
tinystt --model data/tinystt.tstt --grammar "turn (on|off) the (light|fan)" speech.wav
tinystt --model data/tinystt.tstt --words yes,no,stop speech.wav
tinystt --model data/tinystt.tstt --hotword Schatzmann:4 speech.wav
tinystt --model data/tinystt.tstt --beam 32 --lm-weight 0.5 --word-bonus 0 speech.wav   # decoder tuning
sox speech.mp3 -t raw -r 16000 -c 1 -b 16 -e signed - | tinystt --model data/tinystt.tstt -

# the two stages in two processes, connected by a pipe
tinystt --model data/tinystt.tstt --phonemes-only speech.wav | tinystt --model data/tinystt.tstt --from-phonemes
tinystt --model data/tinystt.tstt --text-phonemes speech.wav   # readable phoneme frames
```

`research/eval_wer.py` runs the command on a manifest and computes the word
error rate.

### Tests

| Binary | Covers |
|---|---|
| `test_container` | Model container format, CRC, corruption |
| `test_queue` | Phoneme table, frames, SPSC queue (stress on two tasks), wire protocol, bridge, `PhonemeOutput` |
| `test_features` | Log-mel features with two injected FFTs, Python golden |
| `test_model` | NN ops, int8 kernels, acoustic model f32/int8 vs numpy golden, batching, frame builder |
| `test_budget` | Compute/memory budget of the production shape |
| `test_lexicon` | Lexicon trie, overlay, letter-to-sound, bigram LM vs Python reference |
| `test_decoder` | Beam search, homophones, confusions, word list, hotwords, N-best, grammar |
| `test_pipeline` | End to end with the trained toy model (`research/make_toy_task.py`): word error rate ≤ 0.25. One loop (`STTPipeline`), the `TinySTT` AudioOutput with its decoder task (also restart, final flush, grammar set before `begin()`) and the split over a byte stream must give exactly the same text. |

## What is platform specific

| Topic | Desktop |
|---|---|
| int8 dot product | the portable 4-way unrolled kernel, which the compiler vectorizes (`-O3`, Release build) |
| Weight placement | none needed (weights stay in RAM) |
| Tasks | `std::thread` |

## Results

Production-shape model (0.9 M parameters, random weights: timing only), 40k
words, 5 s of audio, one thread, best of 5 runs (laptop clock scaling makes
single runs noisy).

| | i7-4650U |
|---|---:|
| Features (`AudioRealFFT`, 80 mel) | 0.0008 RTF |
| Acoustic model, int8, batch 4 | 0.58 ms / 30 ms frame |
| **Stage 1** (features + model) | **0.020 RTF** |
| Stage 2 (decoder, 40k words, beam 16) | 0.0001 RTF |

| Model variant | ms per 30 ms frame |
|---|---:|
| float32 | 1.04 |
| int8 weights × float activations | 1.32 |
| int8 activations | 0.56 |
| int8 activations, batch 4 / 8 | 0.58 / 0.59 |

Batching brings nothing on the desktop: the CPU caches hold all the weights.

The accuracy of the shipped model was measured on the desktop; see
[`performance.md`](performance.md#accuracy-and-decoder-tuning).

## Reproducing

```bash
python research/make_bench_model.py --out bench.tstt   # production shape, random weights
./build/desktop/tinystt_benchmark bench.tstt
```
