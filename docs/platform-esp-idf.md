# ESP-IDF

TinySTT is header-only. The top-level `CMakeLists.txt` registers it as an
ESP-IDF component when `ESP_PLATFORM` is set. Add both TinySTT and
arduino-audio-tools as components:

```cmake
# project CMakeLists.txt
set(EXTRA_COMPONENT_DIRS /path/to/arduino-audio-tools /path/to/TinySTT)
```

`main` then `REQUIRES TinySTT arduino-audio-tools`. Without arduino-esp32,
audio-tools runs in its `ESP32_CMAKE` mode. Use `AudioEspressifFFT` (esp-dsp)
for the features.

The model container can live in a data partition and be memory-mapped:
`ModelPartition::begin(model, "tinystt")` does this with `esp_partition_mmap`,
without a RAM copy. See [`model-data.md`](model-data.md) and the partition
setup in [`platform-esp32.md`](platform-esp32.md).
