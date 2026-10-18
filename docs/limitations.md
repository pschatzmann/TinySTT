# Status and limitations

- **Accuracy:** the shipped model reaches 36 % WER on LibriSpeech test-clean
  (beam 16; 33 % at beam 32). That's fine for commands with a grammar, not yet
  for reliable dictation. See [`training.md`](training.md) for the most
  promising improvements (more training data, a larger LM).
- English only (CMUdict phonemes). Other languages need a phoneme set, a
  lexicon and training data.
- The LM is a bigram model from only 18,000 sentences; there is no trigram yet.
- Hotwords get their boost when the word completes; there is no prefix bonus
  yet.
- Confidence is a heuristic (beam posterior × acoustic certainty), not a
  calibrated probability.
- The VAD is energy based; noisy rooms need threshold tuning
  (`vadConfig()`), or `setVADEnabled(false)` with your own endpointing.
- Needs arduino-audio-tools from 2026-10-18 or later (see
  [`requirements.md`](requirements.md)); older releases have the wrong `Hann`
  window, which the trained model is sensitive to.
- RP2350 at 150 MHz: the complete pipeline runs just faster than real time
  (0.97 RTF) only without an OS and with `setInternalRamReserve(48 * 1024)`.
  With the default reserve, after `end()`/`begin()`, or with FreeRTOS it is
  1.2–1.6 × slower than real time (see
  [`platform-rp2350.md`](platform-rp2350.md)). Overclocking was not tested.
