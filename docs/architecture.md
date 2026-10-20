# Architecture

TinySTT is split into two stages that only talk through a queue of
`PhonemeFrame` messages, so they can run in one loop, on two tasks/cores, or on
two devices.

```
           Stage 1: SpeechToPhonemes (AudioOutput)                         Stage 2: PhonemesToText
 PCM ─► 10 ms blocks ─┬─► VAD (VolumeMeter) ── start / end of utterance
 16 kHz               └─► FeatureExtractor ─► AcousticModel ─► FrameBuilder ─►  RingBufferSPSC  ─► CtcBeamDecoder ─► SttResult
 mono                     (injected FFT,      (conv + GRU,     (top-k,           <PhonemeFrame>     (Lexicon trie +    (N-best,
 int16                     log-mel 80)         CTC, 30 ms)      blank runs)            │             LM or Grammar)     confidence)
                                                                                       │
                                     other device: QueueToStream ─► Serial/UDP/ESP-NOW ─► StreamToQueue
```

## Stage 1: Speech to phonemes

| Component | File | What it does |
|---|---|---|
| `VAD` | `src/TinySTT/VAD.h` | audio-tools' `VolumeMeter` on each 10 ms block, with start, hangover and maximum-length logic (after 15 s an utterance ends at its next short pause, after 30 s unconditionally). While silent, only features are computed. The last 30 frames (300 ms) are replayed when speech starts, so the onset is not lost. |
| `FeatureExtractor` | `src/TinySTT/FeatureExtractor.h` | Uses the **caller's** `AudioFFTBase` object (`AudioRealFFT`, `AudioEspressifFFT`, ...). 512-point frames with a 160-sample hop, audio-tools `Hann` window (cached by `BufferedWindow`), then `toMEL(80, 0, 8000, power)`, `ln(x + 1e-6)` and the global mean/std normalization from the container. |
| `AcousticModel` | `src/TinySTT/AcousticModel.h` | Causal conv1d (kernel 5, stride 3) + ReLU → 2 × GRU(256) → linear → log-softmax over 41 classes (blank, silence, 39 ARPAbet phonemes). One output every 30 ms. The default precision is int8 weights with int8 activations, with a per-platform dot-product kernel: esp-nn SIMD (ESP32-S3), PIE SIMD (ESP32-P4), DSP `SMLAD` (RP2350 / Cortex-M33), NEON (ARM Linux), otherwise an unrolled portable loop. Weights can be placed in PSRAM / internal RAM, and n frames can be batched per pass (see [`performance.md`](performance.md)). |
| `FrameBuilder` | `src/TinySTT/FrameBuilder.h` | Keeps the top-k (≤ 4) classes per frame with int8 log-probabilities. It merges near-certain blanks (P > 0.95) into one `BlankRun` message, which removes most of the traffic. |

The production shape has 0.90 M parameters and needs 0.90 M MAC per output
frame (30 M MAC/s): 0.43 × real time on an ESP32-S3, 0.16 × on an ESP32-P4, 0.92 × on an RP2350. `test_budget` checks it against the budget in
[`plan.md`](plan.md).

## The queue

`PhonemeQueue` is audio-tools' lock-free single-producer/single-consumer
`RingBufferSPSC<PhonemeFrame>`. It is the same class on desktop, emulator and
MCU. `PhonemeFrame` is a 16-byte POD (see [`protocol.md`](protocol.md)).

The SPSC buffer never blocks, so `PhonemeQueueWriter` handles a full queue:
- It retries with `delay(1)` up to a timeout (default 100 ms).
- After the timeout it drops the frame and counts it (`droppedFrames()`).
- `EndOfUtterance` and `Reset` are never dropped.

## Stage 2: Phonemes to text

| Component | File | What it does |
|---|---|---|
| `Lexicon` | `src/TinySTT/Lexicon.h` | Pronunciation trie in two parts. The static part comes from the container (CMUdict, flash friendly). A RAM overlay holds the words added at runtime. `LetterToSound` covers unknown words. |
| `LanguageModel` | `src/TinySTT/LanguageModel.h` | Bigram backoff model over the lexicon word ids, int16 log-probabilities, read in place. |
| `Grammar` | `src/TinySTT/Grammar.h` | Command mode: `( a \| b )`, `?` and `;` compile to a word DFA that replaces the LM. |
| `CtcBeamDecoder` | `src/TinySTT/CtcBeamDecoder.h` | Lexicon-constrained CTC prefix beam search. At each node that ends a word, a word-completed hypothesis is scored by the `WordScorer` (LM or grammar), plus a word bonus and hotword boosts. It reports partial results and a final N-best result with confidences. Word histories that no hypothesis uses any more are pruned, so its memory depends on the beam, not on the length of the utterance (about 30 KB at most with the shipped model). |

Decoder score of a hypothesis:

```
log P_ctc(phoneme prefix) + lm_weight · Σ log P(word | history) + word_bonus · #words + Σ boosts
```

Confidence is the posterior of the hypothesis among the final beam, multiplied
by the mean top-1 probability of the non-blank frames. Flat, uncertain
acoustics therefore lower it, even when one path dominates the beam.

## API levels

| Class | Use |
|---|---|
| `TinySTT` | An audio-tools `AudioOutput`: `TinySTT stt(fft, model, Serial)` and `StreamCopy copier(stt, mic)`. `write()` runs speech → phonemes in the caller's task (`loop()`, core 1 on ESP32). `begin()` initializes both stages and starts the decoder task (core 0 on ESP32, core 1 on RP2040/RP2350). On boards without tasks (RP2350 without FreeRTOS) the decoder runs inside `write()` instead. Accepted final results are printed as lines. It expects 16 kHz mono 16 bit (convert in front otherwise). Settings go through `speechToPhonemes()` / `phonemesToText()` before `begin()`; `setTaskConfig()` sets the decoder task's stack, priority and core. |
| `STTPipeline` | Both stages and the queue, with explicit control: `write(pcm)` + `copy()` in one loop, `startDecoderTask()`, or `startTasks(stream)` for both tasks. Used by the desktop CLI and the tests. |
| `SpeechToPhonemes`, `PhonemesToText`, `QueueToStream`, `StreamToQueue` | The individual stages and the bridge, for splitting across devices or processes. |
| `PhonemeOutput` | Prints the phonemes of step 1 in a human readable form, to `Serial` by default. It takes the place of the phoneme queue: `SpeechToPhonemes stp(fft, phonemeOutput)`. Two modes: one line per utterance with the most likely phoneme sequence (greedy CTC), or every frame with its probabilities. |

Grammars and words added before `begin()` are applied in `begin()`, once the
dictionary is loaded, so they get its pronunciations. A grammar's syntax is
checked immediately.

The tasks yield at least every 50 ms, and the decoder task handles at most 16
queue messages per step, so a decoder that falls behind (e.g. a very wide
beam) does not starve the idle task and trip the watchdog.

`end()` stops the tasks cooperatively. Each task finishes its current step
and parks before it is suspended, so it is never stopped inside an
allocation, and `begin()` can start them again. `TinySTT::end()` first sends
the end of the current utterance and waits until the decoder has taken it,
so the last result is printed.

## Threading options

| Setup | How |
|---|---|
| Two tasks / cores | `TinySTT stt(fft, model, Serial); StreamCopy copier(stt, mic); stt.begin();` then `copier.copy()` in `loop()` |
| One loop | `STTPipeline p(fft); p.begin(model);` then `p.write(pcm); p.copy();` |
| Two devices | A: `SpeechToPhonemes` + `QueueToStream`. B: `StreamToQueue` + `PhonemesToText`. |
| Two processes | `tinystt --phonemes-only x.wav \| tinystt --from-phonemes` |

The end-to-end tests check that all of these produce identical text.
