# TinySTT – Plan

## Context
TinySTT is a new Arduino library (only an empty `docs/`). The goal is a speech-to-text library that is the counterpart of TinyTTS:
- It runs on a microcontroller (ESP32-S3 / ESP32-P4 first).
- It can be built and tested on the desktop through arduino-audio-tools' CMake, which fetches pschatzmann/Arduino-Emulator.
- arduino-audio-tools provides the audio streams, Task, Mutex and the synchronized queues and buffers.
- It is split into two independent stages, **Speech→Phonemes** (acoustic) and **Phonemes→Text** (decoder). The stages are connected by a queue of phoneme frames, so they can run on different Tasks or, through a stream bridge, on different devices.

Decisions you made:
- The acoustic model is our own small CTC phoneme model, trained in `research/`. Inference is hand-written C++ with no TFLite.
- Phonemes→Text uses a large vocabulary (CMUdict plus a pruned n-gram LM). You can also restrict it to your own word or command list.
- The two stages pass information through a lock-free single-producer/single-consumer queue (`RingBufferSPSC`). Across devices, a binary protocol over any Arduino `Stream` bridges two such queues.

The first step is an un-optimized, platform-independent version with automatic tests. The second step optimizes for ESP32-S3 and ESP32-P4. `docs/plan.md` holds this plan and tracks implementation state (a checkbox per item).

## Architecture
```
mic/I2S/WAV ─► FeatureExtractor (log-mel 80×10ms) ─► AcousticModel (CNN+GRU, CTC) ─► PhonemeFrame
                                                                                     │ write()  (single producer)
                                            ┌──── RingBufferSPSC<PhonemeFrame> ──────┘
                                            │  same device: shared directly between the two Tasks / cores
                                            │  other device: QueueToStream ─► Serial/UDP/ESP-NOW ─► StreamToQueue
                                            ▼ read()   (single consumer)
text ◄── PhonemesToText (CTC prefix beam search + Lexicon trie + n-gram LM)
```
**The two processing steps exchange information only through a queue of `PhonemeFrame` messages.** A `PhonemeFrame` is a fixed-size POD:
- frame index;
- type (`Frame`, `BlankRun`, `EndOfUtterance`, `Reset`);
- k;
- `top[k]` entries of (phoneme id u8, logprob int8);
- for `BlankRun`, the run length.

**Blank skipping:** consecutive frames with P(blank) above a configurable threshold (default 0.95) are merged into one `BlankRun` frame. The decoder expands a run as repeated blank steps, which is exact for CTC prefix search, and uses it as a word-boundary hint. This cuts decoder steps and bridge traffic about 5-10x. A threshold of 1.0 turns skipping off, and a test checks that both settings decode the same text.

**The queue is audio-tools' lock-free single-producer/single-consumer `RingBufferSPSC<PhonemeFrame>` (`Concurrency/LockFree/RingBufferSPSC.h`).**
- The topology is exactly one producer (`SpeechToPhonemes`) and one consumer (`PhonemesToText`), so SPSC fits with no mutex. Head and tail are atomic and safe across the two ESP32 cores.
- It is the same class on desktop, emulator and MCU, so there are no platform `#if` branches in the transport.
- `PhonemeFrame` is POD and fixed-size, which suits the buffer's `memcpy` element copies. Capacity is rounded to a power of two, for example 64 frames (640 ms). `setUsePSRAM(true)` is optional.
- The stages take a `BaseBuffer<PhonemeFrame>&` (the interface `RingBufferSPSC` implements), so tests can still pass a plain `RingBuffer`. `STTPipeline` and the examples always use `RingBufferSPSC`.
- The SPSC buffer does not block:
  - **Producer when full:** retries with `delay(1)` up to a configurable timeout. After that it drops the frame, counts it in `droppedFrames()`, and logs. This keeps the audio path from stalling forever.
  - **Consumer when empty:** `PhonemesToText::copy()` returns 0, and the Task loop yields with `delay(1)`.
  - `EndOfUtterance` is never dropped: the producer waits until there is room.

For the split across devices, each device keeps its own SPSC queue and a bridge connects them:
- On the sender, `QueueToStream` is the consumer of the local queue. It serializes frames with `PhonemeProtocol` onto any Arduino `Stream`.
- On the receiver, `StreamToQueue` is the producer of the local queue. `PhonemesToText` is therefore identical in both setups, and the SPSC rule holds on each side.
Phoneme set: 39 ARPAbet phonemes plus blank and silence, the same as TinyTTS (`TinyTTS/src/TinyTTS/PhonemeEncoder.h`, `CmuDict.h`), so the tooling and dictionary format can be shared.

## Reuse from arduino-audio-tools (checked in `arduino-audio-tools/src/AudioTools`)
| Need | audio-tools class (file) | Note |
|---|---|---|
| FFT (selectable) | `AudioFFTBase` plus drivers `AudioRealFFT`, `AudioKissFFT`, `AudioEspressifFFT`, `AudioESP32FFT`, `AudioCmsisFFT`, `AudioFixedFFT` (`FFT/`) | Passed in as an object. Framing with hop (`stride`), windowing and int16 scaling are already done in `processSamples()`. |
| Window | `Hann` (`FFT/FFTWindows.h`) | |
| **Mel spectrogram** | `AudioFFTBase::toMEL(n_bins, fmin, fmax)` (`FFT/AudioFFT.h:404`) | This is the only standalone mel implementation. `TfLiteAudioStream`'s spectrogram uses the TFLM micro-frontend, which needs TFLite, so it is **not** used. TinySTT adds only `log()` and normalization on top of `toMEL()`. |
| Format / sample rate to 16 kHz mono int16 | `FormatConverterStream`, `ResampleStream`, `ChannelFormatConverterStream` (`CoreAudio/`) | Any mic or WAV is converted before the features. |
| Pre-emphasis / DC removal | `FilteredStream` + `FIR` / `BiQuad` high-pass (`CoreAudio/AudioFilter/Filter.h`) | Optional, chained in front. |
| Energy for VAD | `VolumeMeter` (`CoreAudio/AudioStreams.h:1665`) | `VAD.h` only adds hysteresis and hangover logic on top. |
| WAV input (tests, CLI) | `WAVDecoder` + `EncodedAudioStream` (`AudioCodecs/`) | No custom WAV parser. |
| Test byte streams | `MemoryStream`, `RingBufferStream` (`CoreAudio/AudioStreams.h`) | Loopback and chunked-read tests. |
| **Queue between the two steps** | `RingBufferSPSC<PhonemeFrame>` (`Concurrency/LockFree/RingBufferSPSC.h`) | Lock-free SPSC with one producer and one consumer. Same class on all platforms. |
| Tasks / locking | `Task`, `Mutex`, `LockGuard` (`Concurrency/`) | |
| Containers / PSRAM | `Vector`, `AllocatorPSRAM` / `AllocatorExt` (`CoreAudio/AudioBasic/Collections/`) | Weights, lexicon and LM in PSRAM on ESP32. |
| Audio I/O | `I2SStream` / `AudioBoardStream` (MCU), `MiniAudioStream` (desktop mic) | |

Not available in audio-tools, so TinySTT implements it: the NN ops and the acoustic model, the phoneme protocol, the lexicon, the LM and the CTC beam decoder.

## Layout (mirrors TinyTTS)
- `library.properties`, `idf_component.yml`, `CMakeLists.txt`: header-only INTERFACE lib `TinySTT::TinySTT`, ESP-IDF branch, options `TINYSTT_BUILD_TESTS` and `TINYSTT_BUILD_DESKTOP`. The arduino-audio-tools FetchContent pattern is copied from `TinyTTS/CMakeLists.txt`. Desktop/test builds use `ADD_ARDUINO_EMULATOR ON` (the Arduino-Emulator), plus `USE_STD_CONCURRENCY` / `USE_CPP_TASK` for the `Desktop/Task.h` and `Mutex.h` paths.
- `src/TinySTT.h`: facade `TinySTT(FFT, ModelContainer, output Print)`, an audio-tools `AudioOutput` fed by a `StreamCopy` in the sketch. `begin()` starts both stages and the decoder task, and results are printed as lines (simplified API, 2026-10-18).
- `src/TinySTT/`:
  - `Config.h`, `Phonemes.h`: phoneme table and ids.
  - `Mat.h` / `Ops.h`: plain float ops (conv1d, GRU, linear, log-softmax). These follow TinyTTS `Mat.h`/`Ops.h`, adapted rather than depended on.
  - `FeatureExtractor.h`: built entirely on the arduino-audio-tools FFT API (`AudioTools/FFT/AudioFFT.h`).
    - **You choose the FFT by passing it in as an object.** `FeatureExtractor(AudioFFTBase& fft)`, and the public stage and facade forward it: `SpeechToPhonemes stt(fft, out)` and `TinySTT stt(fft, model, Serial)`. The library never instantiates an FFT driver itself.
    - The sketch or app picks any `AudioFFTBase` subclass: `AudioRealFFT` (portable, used by the tests and the desktop CLI), `AudioKissFFT`, `AudioEspressifFFT` / `AudioESP32FFT` (ESP32), or `AudioCmsisFFT`.
    - The extractor calls `fft.begin(cfg)` with its own config and installs its callback through `cfg.ref`. The FFT object passed in must therefore be dedicated to the extractor.
    - Config: 16 kHz mono int16, `length=512`, `stride=160` (10 ms hop), `window_function=new Hann()`.
    - On each `callback`, `fft.toMEL(80, 0, 8000)` gives the mel bins, followed by `log(x+eps)` and streaming mean normalization. This pushes one 80-dim frame into the acoustic model.
    - `research/features.py` re-implements exactly this pipeline (same Hann, int16 scaling, `toMEL` triangular weights) so training features match the device.
    - **Prerequisite: an upstream fix of `toMEL()` in arduino-audio-tools** (separate PR, Phase 1 step 0). Today's code at `FFT/AudioFFT.h:404` has two problems:
      - it allocates three `Vector`s and runs `pow`/`log10` on every call;
      - it rounds filter edges to whole bins, so the lowest of 80 filters at 512/16 kHz have zero width and always output 0.

      The PR changes `toMEL()` in a backward-compatible way:
      - the filterbank is cached and rebuilt only when (n_bins, fmin, fmax, length, rate) change;
      - fractional-bin (HTK/librosa-style) triangle weights, so no filter has zero width;
      - an optional `power` flag (|X|² instead of |X|).

      Until the PR is merged, TinySTT's CMake pins a branch or commit that contains it.
  - `AcousticModel.h`: streaming causal model.
    - Structure: conv front-end with **stride 3 (30 ms output frames)**, then 2-3 GRU layers (or causal TDNN), then a linear layer to 41 classes, then log-softmax.
    - Layer widths are multiples of 16, ready for SIMD and int8.
    - Weights come from the model container.
  - `ModelContainer.h`: one versioned binary format for weights, lexicon, LM and grammar.
    - Contents: magic, version, section table, per-section dtype (f32 / int8 + scales), CRC32.
    - It is read zero-copy from a PROGMEM array, from a file (`File`/FS, which loads into PSRAM), or on ESP32 from a memory-mapped flash data partition (`esp_partition_mmap`).
    - `research/` writes it, and the C++ checks version and CRC in `begin()`.
  - `PhonemeFrame.h`: the POD message type placed in the queue.
  - `SpeechToPhonemes.h`: an `AudioOutput` sink (write PCM in). The constructor is `SpeechToPhonemes(AudioFFTBase& fft, BaseBuffer<PhonemeFrame>& queue)`. It is the only producer and writes `PhonemeFrame`s, plus `EndOfUtterance` from the VAD.
  - `QueueBridge.h`: `QueueToStream` (consumer side) and `StreamToQueue` (producer side) for the split across devices or processes.
  - `PhonemeOutput.h`: prints step 1's phonemes in a readable form (phoneme sequence per utterance, or every frame), in place of a queue.
  - `PhonemeProtocol.h`: wire format, used only by the bridge. Each frame is a sync byte, seq, frame index, k, then k×(id u8, logprob int8 quantized). End-of-utterance and reset frames, CRC8. A text debug mode prints ARPAbet.
  - `Lexicon.h`: CMUdict → phoneme trie, loadable from a header or a file. A user word list is turned into phonemes with the dictionary, plus a simple rule-based letter-to-sound fallback.
  - `LanguageModel.h`: compact quantized ARPA-derived bigram/trigram, binary format. A uniform LM is used when none is loaded.
  - `CtcBeamDecoder.h`: lexicon-constrained CTC prefix beam search with LM fusion and a word insertion bonus. It is streaming: partial results plus a final result. It also offers:
    - **Hotword / context biasing:** `addHotword("Schatzmann", boost)` adds or boosts words in the lexicon trie and gives a score bonus when the beam enters their prefix, with no retraining. Unknown hotwords get phonemes from the letter-to-sound fallback.
    - **N-best + confidence:** the final result holds the top-N hypotheses with total score and per-word confidence (a posterior from the softmax over beam scores, plus average acoustic frame probability). `setMinConfidence()` rejects results below the threshold.
  - `Grammar.h`: **command mode.** A small grammar syntax, e.g. `turn (on|off) the (kitchen|bedroom)? (light|fan)`, compiles to a word-level finite-state graph. The graph constrains the beam search in place of the n-gram LM. You select free (lexicon + LM), word-list or grammar mode at runtime. The grammar can be loaded from a string or from the model container.
  - `PhonemesToText.h`: takes a `BaseBuffer<PhonemeFrame>&` and is its only consumer. It reads `PhonemeFrame`s and emits partial and final `SttResult`s (N-best, confidence, timings) through a callback, or the best text to a `Print`.
  - `VAD.h`: endpointing on top of audio-tools' `VolumeMeter`, with thresholds and hangover. It produces end-of-utterance.
  - `STTPipeline.h`: `STTPipeline` (renamed from `Pipeline`, which clashes with `audio_tools::Pipeline`) owns one `RingBufferSPSC<PhonemeFrame>`. It offers `write()`/`copy()` for one loop or `startTasks()`: the producer task (audio + acoustic model) and the consumer task (decoder), pinned to different cores on ESP32.
  - `data/`: generated headers (`default_acoustic_weights.h`, `default_lexicon_slim.h`, `default_lm.h`).
- `research/` (Python): data prep (LibriSpeech plus CMUdict alignment to ARPAbet targets), training of the CTC model (PyTorch), export of weights, golden test vectors (features, logits, decoded text), and LM build from LibriSpeech LM text. A Dockerfile and docs.
- `test/`: `CMakeLists.txt` plus a minimal assert-based harness (`test/TestUtil.h`, no external framework, as in TinyTTS). One executable per area, registered with CTest.
- `desktop/`: `tinystt` CLI. It reads WAV or a mic (MiniAudioStream) and prints text. A `--phonemes-only` flag prints the phoneme stream to stdout and `--from-phonemes` reads it from stdin, which shows the split between processes through a Unix pipe.
- `examples/`: `stt_i2s_mic` (single device, two tasks), `stt_split_sender` / `stt_split_receiver` (two devices over Serial / ESP-NOW), `timing_benchmark`.
- `docs/`: `plan.md` (status tracker), `architecture.md`, `phonemes.md`, `protocol.md`, `model-data.md`, `requirements.md`, `platform-desktop.md`, `platform-esp-idf.md`, `training.md`, `research.md`, `performance.md`, `limitations.md`.
- `README.md` uses TinyTTS's structure: badges, intro/status, sample, how it works, documentation table, conclusions with a real-hardware table, attribution, license.

## Compute / memory budget (design target, checked by tests)
Target: real time on ESP32-S3 (240 MHz, 2 cores, 8 MB PSRAM, 16 MB flash) after Phase 2. On ESP32-P4 the target is under 0.5x real time.

| Stage | Budget |
|---|---|
| Features (FFT 512 + mel 80, 100 frames/s) | ≤ 15% of one core |
| Acoustic model (33 frames/s) | ≤ 1.5 M params, ≤ 1.5 M MAC/frame (about 50 M MAC/s), ≤ 1.5 MB int8 weights, ≤ 64 KB activations + state in internal RAM |
| Decoder (core 0) | beam ≤ 16, ≤ 300 KB PSRAM working set, ≤ 1 core at blank-skipped frame rate |
| Data in flash | weights ≤ 1.5 MB + lexicon ≤ 2.5 MB + LM ≤ 4 MB, total ≤ 8 MB (container in a flash partition) |
| Queue | 64 × `sizeof(PhonemeFrame)` (k=4, about 16 B), about 1 KB |

`test_budget` loads the model container and computes parameter count, MACs per frame and peak activation size from the layer table, plus the sizes of the lexicon, LM and container. It fails when a limit is exceeded. Phase 2 adds the measured time per stage on hardware.

## Test audio
- **TinyTTS-generated corpus:** `research/make_test_audio.py`, or a small C++ tool linking the sibling TinyTTS (used through its desktop CLI), synthesizes deterministic 16 kHz WAVs of fixed phrases, commands and hotwords into `test/data/`. These are short clips, so the repo stays small.
- **TTS→STT round-trip test** (`test_roundtrip`, optional, enabled when TinyTTS is found next to TinySTT or fetched): it synthesizes text with TinyTTS at runtime, transcribes it with TinySTT, and asserts WER ≤ threshold and a phoneme error rate ≤ threshold against TinyTTS's own phoneme sequence. Both libraries share the ARPAbet set, so this also checks phoneme alignment directly.
- Real-speech WER on a LibriSpeech test-clean subset stays in `research/eval.py`, which is not run by ctest because of data size. Results are recorded in `docs/plan.md`.

## Phase 1 – platform-independent, un-optimized (each step has its tests)
0. Upstream PR to arduino-audio-tools fixing `toMEL()`: cached filterbank, fractional weights, `power` flag. It ships with its own unit test in audio-tools.
1. Scaffolding: CMake, library.properties, idf_component, test harness, CTest, `docs/plan.md`, `ModelContainer` reader/writer with round-trip, CRC and version tests, `test_budget`.
2. Phoneme table, `PhonemeFrame`, queue integration and the bridge. Tests:
   - `RingBufferSPSC<PhonemeFrame>` write/read order and EndOfUtterance;
   - the full-queue timeout and drop counter, and that EndOfUtterance is never dropped;
   - a stress test with the producer and consumer on two `Task`s (std::thread on desktop), 100k frames, no loss or reordering, and a clean run under `-fsanitize=thread`;
   - `QueueToStream` → `MemoryStream` → `StreamToQueue` round trip;
   - resync after corrupt bytes, CRC, chunked reads.
3. `FeatureExtractor` with the FFT injected. The tests run the same suite once each with `AudioRealFFT` and `AudioKissFFT` passed in, to prove the selection works. Tests:
   - a sine's mel energy peaks at the expected bin;
   - frame count and hop match `stride`;
   - golden log-mel vectors from `research/features.py` match (mean abs diff < 1e-4, max < 5e-3: float32 device FFT vs float64 numpy);
   - writing in arbitrary chunk sizes gives the same frames as writing at once.
4. Ops and `AcousticModel` (float plus a scalar int8 reference path). Tests:
   - each op against small numpy-generated goldens;
   - full model logits (f32 and int8) against PyTorch goldens;
   - the stride-3 output frame count;
   - streaming chunk invariance;
   - blank-run generation.
5. `research/` training with quantization-aware training (fake-quant int8 for the last epochs), exported as both f32 and int8 into the model container. A tiny placeholder model with random weights keeps tests running before training finishes.
6. `Lexicon` trie and letter-to-sound fallback. Tests: lookup, homophones, OOV fallback, user word list.
7. `LanguageModel`. Tests: probabilities against a reference ARPA, backoff, binary load.
8. `CtcBeamDecoder` (fed from a queue of synthetic frames). Tests:
   - synthetic posteriors from known phoneme sequences decode to the exact words;
   - noise and substitution robustness;
   - word-list mode;
   - partial vs final results;
   - with blank skipping on and off, the same result;
   - a hotword boost flips an ambiguous homophone to the biased word;
   - N-best ordering, and that confidence drops on noisy input;
   - `setMinConfidence` rejects low-confidence results.
8b. `Grammar` compile and grammar-constrained decoding. Tests: parsing (alternatives, optional, groups), out-of-grammar input is rejected or low-confidence, all grammar paths decode from synthetic posteriors.
9. `SpeechToPhonemes` / `PhonemesToText` / `STTPipeline`. Tests: end-to-end on the TinyTTS-generated clips (free, grammar and hotword modes), with a WER threshold, `test_roundtrip`, single-threaded (same queue, drained in a loop), two-Task over the shared queue, and split through queue → bridge → loopback stream → bridge → queue with injected chunking. All three must give identical text.
10. Desktop CLI, examples, docs, and the README.

## Phase 2 – ESP32-S3 / ESP32-P4 optimization
- Switch the default to the int8 weights already produced by quantization-aware training in Phase 1. The scalar int8 path from Phase 1 is the reference for the SIMD kernels.
- Flash-mmap the model container partition on ESP32, so no data is copied to RAM.
- SIMD dot products: ESP32-S3 via the esp-nn / esp-dsp `dsps_dotprod` path, reusing the approach of `TinyTTS/src/int8-dotprod`; ESP32-P4 via its PIE SIMD.
- The ESP32 examples pass an `AudioEspressifFFT` object (esp-dsp) instead of `AudioRealFFT`. Library code does not change. A test checks the features stay within tolerance of the `AudioRealFFT` result.
- PSRAM placement of the lexicon and LM, an IRAM hot loop, `SpeechToPhonemes` pinned to core 1 and the decoder to core 0.
- Beam width and top-k tuning.
- Real-time factor measurements in `docs/performance.md`.
- Each optimization keeps the host tests green: the optimized kernels also have scalar reference implementations, compared in tests.

## Verification
- `cmake -S . -B build -DTINYSTT_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build` runs all unit and end-to-end tests on the desktop through the Arduino-Emulator / audio-tools.
- `./build/desktop/tinystt test/data/hello.wav` prints the transcript. `tinystt --phonemes-only x.wav | tinystt --from-phonemes` gives the same text.
- Arduino CLI compile of the examples for esp32s3, esp32p4 and rp2350. A hardware run of `timing_benchmark` records the real-time factor and WER in `docs/plan.md` and `docs/performance.md`.

## Implementation status
Legend: `[ ]` open · `[~]` in progress · `[x]` done (tests green)

### Phase 1 – platform-independent
| # | Step | State | Tests | Notes |
|---|---|---|---|---|
| 0 | Upstream `toMEL()` fix in arduino-audio-tools | [x] | `tests-cmake/fft/mel`, `tests-cmake/codec/wav-decoder-chunks` PASS | Merged into audio-tools `main` as PR #2445 (2026-10-18). Changes:
  - mel filterbank, `toMEL` power via `magnitudeFast`;
  - `reset()` stride buffer + `resize(0)` log;
  - `WAVDecoder` header/data loss;
  - FFT driver leak;
  - `BufferedWindow` empty-cache crash;
  - faster mono framing + `toMEL`;
  - review fixes: RTOS `Task` creation/null handle, `VolumeMeter` average, 24-bit FFT input, partial-sample writes, magnitude buffer size;
  - `Hann` window formula (0.54 → 0.5). | |
| 1 | Scaffolding, `ModelContainer`, `test_budget`, `docs/plan.md` | [x] | `test_container` (6), `test_budget` (3) green | CMake prefers the sibling audio-tools checkout; `-DFETCHCONTENT_SOURCE_DIR_ARDUINO_EMULATOR=<path>` uses a local Arduino-Emulator. Production shape: 0.90 M params, 0.90 M MAC/frame (30 M MAC/s), 12.6 KB activations. |
| 2 | Phonemes, `PhonemeFrame`, SPSC queue, bridge + protocol | [x] | `test_queue` (11) green, TSan clean | TSan on Linux 6.x needs `setarch $(uname -m) -R` (ASLR). |
| 3 | `FeatureExtractor` (injected FFT) | [x] | `test_features` (11) green | Second injected FFT is a test-only naive DFT (`test/NaiveDFT.h`): the Arduino kissfft build is 16-bit fixed point, so `AudioKissFFT` is not comparable. audio-tools' `Hann` window (0.5·(1−cos) since the 2026-10-18 fix; it was 0.54·(1−cos)) is replicated in `research/features.py`. |
| 4 | Ops + `AcousticModel` (f32 + int8 ref) | [x] | `test_model` (9) green | conv(k5,s3)+ReLU → 2×GRU(256) → linear → log-softmax. C++ vs numpy reference (`research/model_numpy.py`): f32 1.2e-6, int8 9.5e-7. `FrameBuilder` does top-k + blank runs. Placeholder model: `research/make_placeholder_model.py`. |
| 5 | `research/` training (QAT) + export | [x] | Kaggle GPU run v11 (460 h): dev PER 15.2 %, test PER 15.0 % | `train.py` (CTC, SpecAugment, speed/noise aug, QAT via fake-quant + `torch.func.functional_call`, export with torch-vs-numpy check), `prepare_librispeech.py`, `eval.py`, `pack.py`, Dockerfile, `kaggle_train.ipynb`. Toy task (phoneme tones) trains to PER 0.000 incl. int8 export. LibriSpeech training on Kaggle (GPU T4) produced `data/tinystt.tstt`; test-clean WER 36.0 % (beam 16) / 32.7 % (beam 32). |
| 6 | `Lexicon` + letter-to-sound | [x] | `test_lexicon` (lexicon part) green | Static flash trie (`build_lexicon.py`) + RAM overlay for user words/hotwords; word-list mode = overlay only. Test data: `research/make_test_data.py` (embedded mini dictionary, no download). |
| 7 | `LanguageModel` | [x] | `test_lexicon` (LM part) green | Bigram backoff with absolute discounting, int16 log values (`build_lm.py`); matches the Python reference to 1e-4, normalized to ±2%. Trigram not implemented (bigram fits the flash budget; revisit after real WER numbers). |
| 8 | `CtcBeamDecoder` (hotwords, N-best, confidence) | [x] | `test_decoder` (10 decoder tests) green | Lexicon-constrained CTC prefix beam search, `WordScorer` interface (LM or grammar). 12/12 sentences correct with 15% confused frames. Confidence = beam posterior × mean top-1 frame probability. Hotword boost is applied on word completion (no prefix bonus yet). Fixed: overlay ids collided with LM specials. |
| 8b | `Grammar` command mode | [x] | `test_decoder` (3 grammar tests) green | `( a \| b )`, `?`, `;`/newline → NFA → DFA; scores −log(branching). Grammar words are added to the lexicon overlay. |
| 9 | `SpeechToPhonemes` / `PhonemesToText` / `STTPipeline` + end-to-end | [x] | `test_pipeline` (11) green (toy WER 0.083, all setups identical); `test_roundtrip` (optional) | End-to-end with a *trained* toy model (`research/make_toy_task.py`: phoneme tones, retrained after the Hann fix with 440 utterances and 48 units). Accuracy is checked as a word error rate ≤ 0.25. One loop, the TinySTT facade on two tasks, restart, final flush and the split over a fragmented byte stream must give exactly the same text. Also covered: grammar mode, VAD segmentation, silence. TTS→STT round trip (`-DTINYSTT_ROUNDTRIP=ON`, `TINYSTT_MODEL=<container>`) runs with the trained model: WER 0.20 on 4 synthesized sentences (3 exact). Found and fixed upstream (merged into audio-tools `main`): `WAVDecoder` dropped the audio that arrived in the same `write()` as the header (losing 250 ms of leading audio changed the CLI result). |
| 10 | Desktop CLI, examples, docs, README | [x] | all 9 test suites green, also under ASan; TSan clean (queue, TinySTT facade) | `tinystt` CLI and `tinystt_benchmark` (WAV/raw stdin, N-best, grammar, word list, hotwords, `--phonemes-only`/`--from-phonemes` pipe split, optional mic). Examples: `stt_i2s_mic`, `stt_commands`, `stt_phonemes`, `stt_split_sender`/`_receiver`, `timing_benchmark`, `selftest`: compile for ESP32-S3, ESP32-P4 (arduino-cli, esp32 core 3.3.10) and RP2350 (Arduino-Pico, with and without FreeRTOS). Docs + README (TinyTTS structure). ASan found an upstream leak (FFT driver never deleted), fixed on the audio-tools branch. |

### Phase 2 – ESP32-S3 / ESP32-P4
| Item | State | Notes |
|---|---|---|
| int8 default weights | [x] | `Precision::Auto` = int8 weights + per-vector int8 activations (`Int8Activations`), matched by `research/model_numpy.py` (4.8e-7) |
| SIMD dot products (S3 / P4) | [x] | S3: esp-nn `esp_nn_dot_s8_aligned_esp32s3` (vendored, verified on hardware). P4: own PIE kernel `dotS8P4` (0.26 ms vs 2.48 ms portable for 768×256), self-test PASS. Portable fallback: 4-way unrolled `-O3`. |
| Flash-mmap model container | [x] | `ModelPartition` (`esp_partition_mmap`); benchmark container in partition `tinystt` |
| Weight placement PSRAM / internal | [x] | `Placement::Auto`: PSRAM + GRU recurrent matrices in internal RAM (reserve 96 KB); flash 27 MB/s vs PSRAM 59 MB/s vs internal 643 MB/s on S3 |
| Time batching | [x] | `setBatchFrames(n)`, bit-identical to n=1 (test), examples use 4 |
| `AudioEspressifFFT` in examples | [x] (decided against) | `AudioRealFFT` measured faster on S3 and P4; examples use it |
| Feature extraction speed | [x] | audio-tools: `BufferedWindow` fix, mono framing, `toMEL` single pass + 40 ms first-call fix; warm-up in `begin()` |
| On-device self-test | [x] | `examples/selftest`: all PASS on ESP32-S3 (esp-nn) and ESP32-P4 (PIE); TinySTT facade (StreamCopy, decoder task, end/begin x3) verified on the S3 |
| PSRAM / IRAM placement, core pinning | [x] | placement above; `TinySTT::begin()` / `STTPipeline::startTasks()` pin audio+model to core 1, decoder to core 0 |
| Simplified API | [x] | `TinySTT(fft, model, Print)` is an `AudioOutput` (`StreamCopy copier(stt, mic)`); `begin()` starts the decoder task; grammar/words set before `begin()` are applied with the dictionary; tasks stop cooperatively and can be restarted; tests `tinystt_facade_*` |
| audio-tools code review | [x] | Fixed in audio-tools (with regression tests in `tests-cmake/fft/mel`):
  - RTOS `Task` creation result / null handle;
  - `VolumeMeter` average;
  - 24-bit FFT input crash;
  - partial-sample writes;
  - magnitude buffer size;
  - `Hann` window formula. |
| Desktop benchmark | [x] | `tinystt_benchmark`: stage 1 0.020 RTF on an i7-4650U |
| `PhonemeOutput` | [x] | Human readable output of step 1 (default `Serial`): phoneme sequence per utterance or every frame; `examples/stt_phonemes`; tests in `test_queue` / `test_pipeline`; verified on the ESP32-S3 |
| Beam / top-k tuning | [x] (default beam 32 since 2026-10-18) | `word_end_beam` (word-boundary quota), LM weight 0.5, word bonus 0: test-clean WER 61.9 % → 31.9 % (100 utt.); full test-clean 36.0 % (beam 16) / 32.7 % (beam 32) |
| Trained model | [x] | Kaggle v11 (2026-10-20): train-clean-100 + 360 with the LibriSpeech lexicon, 5 + 1 QAT epochs in 9.6 h, dev PER 15.2 %, test PER 15.0 %; WER 29.6 % (`data/tinystt.tstt`, beam 32), 25.7 % (`data/tinystt_large.tstt`, beam 64). Previously v6: train-clean-100, dev PER 17.3 %, WER 32.7 % |
| RP2350 | [x] | Cortex-M33 DSP kernel, placement (384 KB internal + PSRAM), no-task fallback / FreeRTOS task; stage 1 0.92 RTF at 150 MHz; self-test PASS; `docs/platform-rp2350.md`. Hang fixed (model buffers allocated before the weight placement; decoder word history pruned) and verified with the trained model: complete pipeline 0.97 RTF without OS and with a 48 KB reserve, 1.57 RTF with FreeRTOS; decoder task on core 1 |
| Decoder memory and tasks | [x] | Word history pruned (bounded by the beam, `decoder_history_stays_bounded`); decoder task yields at least every 50 ms and processes at most 16 messages per step (a decoder slower than real time triggered the ESP32 task watchdog); FreeRTOS stack depth in words outside ESP-IDF; RP2040 decoder task on core 1 |
| WER improvements | [~] | Default beam 32 (36.0 % → 32.7 %). Larger lexicon + LibriSpeech LM (`arpa_to_lm.py`): 9.5 MB container 24.7 % at beam 64 (100 utt.), 29.8 % on full test-clean (vs 32.7 %), runs on S3/P4 from flash. N-best trigram rescoring (SD idea): only −0.8 points, not pursued. VAD: no WER cost (an earlier 3-point loss was a measurement error); long utterances now end at a pause (`split_pause_ms`, hard limit 30 s). Retraining on train-clean-100 + 360 with the LibriSpeech lexicon (Kaggle v11): 32.7 % → 29.6 %, with the large container 25.7 %. Next: more epochs (`--resume`) |
| Waveshare ESP32-P4-WIFI6 | [x] | `examples/stt_waveshare_p4`: `AudioBoardStream` (ES8311, onboard microphone read as 16 kHz mono), model compiled in as PROGMEM array (header created on request with `research/export_header.py`, not shipped; 13 MB app partition); recognizes speech on the board. `AudioRealFFT` 0.040 RTF vs `AudioEspressifFFT` 0.052 (and different features), so `AudioRealFFT` stays. Fixed on the way: decoder task start race (TinySTT waits until a task is really suspended before replacing its code: crash on the P4); audio-tools `ChannelFormatConverterStream` used as output wrote nothing (fixed on audio-tools `main`, with test) |
| Raspberry Pi | [x] | NEON kernel, `TINYSTT_NATIVE` (`-mcpu=native`); Pi 4 stage 1 0.016 RTF, same WER as x86; `docs/platform-raspberry-pi.md` |

## Measurements
| Date | Platform | Model | WER | Real-time factor | Notes |
|---|---|---|---|---|---|
| 2026-10-18 | ESP32-S3 240 MHz | production shape, random weights | – | stage 1 0.515, decoder 0.016 | int8 SIMD, PSRAM + 192 KB internal, batch 4 (`timing_benchmark`) |
| 2026-10-18 | ESP32-P4 400 MHz rev 3.2 | production shape, random weights | – | stage 1 0.490, decoder 0.002 | portable unrolled int8 (-O3), batch 4 |
| 2026-10-18 | ESP32-P4 400 MHz rev 3.2 | production shape, random weights | – | stage 1 0.157, decoder 0.002 | PIE SIMD int8, PSRAM + 192 KB internal, batch 4 |
| 2026-10-18 | ESP32-S3 240 MHz | production shape, random weights | – | stage 1 0.434, decoder 0.014 | after the faster features (0.087 RTF) |
| 2026-10-18 | Desktop i7-4650U, 1 thread | production shape, random weights | – | stage 1 0.020, decoder 0.0001 | `tinystt_benchmark`, best of 5 |
| 2026-10-18 | x86 desktop | trained (Kaggle v6) | 36.0 % (beam 16), 32.7 % (beam 32) | – | LibriSpeech test-clean, 2,620 utterances |
| 2026-10-18 | RP2350 150 MHz, 8 MB PSRAM | production shape, random weights | – | stage 1 0.924, decoder 0.004 | DSP SMLAD, 384 KB internal + PSRAM, batch 4 |
| 2026-10-18 | x86 desktop | trained (Kaggle v6) | 33 % (4 sentences) | – | TinyTTS → TinySTT round trip (`test_roundtrip`) |
| 2026-10-18 | ESP32-S3 240 MHz | trained (Kaggle v6) | same transcript as x86 | complete pipeline 0.44 | `TinySTT`, 10.4 s LibriSpeech utterance, model in the flash partition |
| 2026-10-18 | ESP32-P4 400 MHz rev 3.2 | trained (Kaggle v6) | same transcript as x86 | complete pipeline 0.16 | `TinySTT`, 10.4 s LibriSpeech utterance, model in the flash partition |
| 2026-10-18 | RP2350 150 MHz, 8 MB PSRAM | trained (Kaggle v6) | same transcript as x86 | complete pipeline 0.97 (no OS, reserve 48 KB), 1.23 (default reserve), 1.57 (FreeRTOS) | `TinySTT`, 10.4 s LibriSpeech utterance |
| 2026-10-18 | x86 desktop | trained (v6) + 50 k lexicon + LibriSpeech bigram (9.5 MB) | 24.7 % (100 utt., beam 64) | – | `research/eval_wer.py` |
| 2026-10-18 | ESP32-S3 / ESP32-P4 | trained (v6) + 9.5 MB container | improved transcript | complete pipeline beam 64: 0.51 / 0.16 | in place from the flash partition |
| 2026-10-18 | Raspberry Pi 4, 32-bit OS | production shape / trained | 31.9 % (100 utt., same as x86) | stage 1 0.016, decoder 0.0022 | NEON, `-mcpu=native` |
| 2026-10-20 | x86 desktop | trained (Kaggle v11) | 20 % (4 sentences, 3 exact) | – | TinyTTS → TinySTT round trip (`test_roundtrip`) |
| 2026-10-20 | ESP32-P4 (Waveshare ESP32-P4-WIFI6) | – | – | features 0.040 RTF (`AudioRealFFT`) vs 0.052 (`AudioEspressifFFT`) | `stt_waveshare_p4`: onboard microphone, model as PROGMEM array; decoder task start race fixed |
| 2026-10-20 | x86 desktop | trained (Kaggle v11, 460 h) | 29.6 % (beam 32), 25.7 % (large container, beam 64) | – | LibriSpeech test-clean, 2,620 utterances |
