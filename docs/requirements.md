# Requirements

## Libraries

- [arduino-audio-tools](https://github.com/pschatzmann/arduino-audio-tools):
  audio streams, FFT, `VolumeMeter`, `RingBufferSPSC`, `Task`.
- TinySTT needs audio-tools from 2026-10-18 or later (`main` with PR #2445):
  - cached fractional mel filterbank, `toMEL(..., power)`, faster mono framing;
  - the `Hann` window formula (0.5·(1−cos); the trained models depend on it);
  - fixes: `AudioFFTBase::reset()`, 24-bit input, partial-sample writes, the driver leak, `BufferedWindow`, `WAVDecoder` header/data, RTOS `Task` creation, `VolumeMeter` averages.

## Boards

Each platform has its own document with board settings, the platform-specific
code and the measured results:

| Board | Status | Document |
|---|---|---|
| ESP32-S3 with PSRAM | Faster than real time (stage 1 0.43 RTF) | [`platform-esp32.md`](platform-esp32.md) |
| ESP32-P4 | Faster than real time (stage 1 0.16 RTF) | [`platform-esp32.md`](platform-esp32.md) |
| RP2350 with PSRAM | Faster than real time at 150 MHz (stage 1 0.92 RTF) | [`platform-rp2350.md`](platform-rp2350.md) |
| Raspberry Pi (Linux) | Desktop build, NEON (Pi 4: stage 1 0.016 RTF) | [`platform-raspberry-pi.md`](platform-raspberry-pi.md) |
| Desktop (Linux/macOS) | Tests, `tinystt` command, benchmark | [`platform-desktop.md`](platform-desktop.md) |
| RP2040 | Too slow for this model (no FPU, no DSP instructions) | – |
| Other 32-bit boards with ≥ 0.5 MB RAM | Portable code, untested; use `AudioRealFFT` | – |

## Flash

| Part | Size |
|---|---:|
| Program (examples) | 0.1 – 0.2 MB (RP2350), 0.4 – 0.55 MB (ESP32); plus the model if it is embedded (see [`model-embedding.md`](model-embedding.md)) |
| Acoustic model (int8) | 0.93 MB |
| Lexicon (25,814 words) | 1.42 MB |
| Language model (bigram) | 0.78 MB |
| **Total for free dictation (default model `data/tinystt.tstt`)** | **≈ 3.7 MB** |
| *Large model `data/tinystt_large.tstt` instead:* 50k-word lexicon (2.6 MB) + LibriSpeech bigram LM (6.0 MB) | 9.5 MB with the acoustic model; total ≈ 10 MB |

With a command grammar instead of free dictation, the language model is not
needed. A split sender needs only the acoustic model, and the receiver only
lexicon and LM.

## RAM

| Part | Size |
|---|---:|
| Acoustic model state and buffers | 14 KB (36 KB with batches of 4) |
| Features (512-point FFT, 80 mel bands) | ≈ 8 KB |
| Phoneme queue (64 frames) | 1 KB |
| Decoder (beam 32, word history) | up to ≈ 30 KB, plus words added at runtime |
| *Optional, for speed:* GRU recurrent weights in internal RAM | 192 KB with the default reserve (96 KB internal RAM stay free); 384 KB on the RP2350 with `setInternalRamReserve(48 * 1024)` |
| *Optional, for speed:* remaining int8 weights in PSRAM | 0.5 – 0.9 MB |
| Model container | none if used in place from flash (ESP32 partition, embedded model); 3.1 MB (large model: 9.5 MB) PSRAM if loaded from a file system or downloaded |

Real time on a microcontroller needs about 0.5 MB internal RAM and PSRAM
(ESP32-S3, ESP32-P4, RP2350 with PSRAM). Without PSRAM the weights are read
from flash, which works but is slower. `Placement::Auto` decides what goes
to internal RAM and PSRAM (see [`performance.md`](performance.md)).
