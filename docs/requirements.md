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

## Memory

Production shape:
- acoustic model state about 14 KB (36 KB with batches of 4);
- features (512-point FFT, 80 mel bands) about 8 KB;
- queue 1 KB (64 frames);
- decoder up to about 30 KB (beam and word histories), plus words added at runtime;
- optional, with `Placement::Auto`: the GRU recurrent matrices in internal RAM (192 KB with the default reserve of 96 KB free internal RAM; 384 KB on the RP2350 with `setInternalRamReserve(48 * 1024)`) and the remaining weights in PSRAM;
- the model container (3.1 MB with lexicon and LM, 9.5 MB for the large container): used in place from flash, or loaded into PSRAM.
