# dsp (vendored)

- **`esp_nn_dot_s8_esp32s3.S`**: ESP32-S3 SIMD int8 dot product
  (`esp_nn_dot_s8_aligned_esp32s3`) from
  [espressif/esp-nn](https://github.com/espressif/esp-nn), Apache-2.0,
  copied from TinyTTS (`src/int8-dotprod`). It requires both pointers to be
  16-byte aligned and `len` a multiple of 16. The file builds to nothing on
  other targets.

The ESP32-P4 PIE kernel (`dotS8P4()` in `DotProduct.h`) is TinySTT's own
inline-assembly implementation. It uses the instruction sequence of
Espressif's precompiled esp-nn `esp_nn_fully_connected_s8_esp32p4` (seen in
its disassembly): `csrsi 0x7f2,1` and `esp.movx.w.cfg 2` for setup,
`esp.vld.128.ip`, `esp.vmulas.s8.xacc[.ld.ip]` and `esp.movx.r.xacc.l`.

The portable reference kernels and the dispatcher are in `DotProduct.h`.
esp-dsp's `dsps_dp_s8_aes3` is deliberately not used: TinyTTS found it
returning wrong results on real ESP32-S3 hardware.
