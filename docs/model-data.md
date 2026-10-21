# Model data

All data lives in one versioned **model container** (`.tstt`,
`src/TinySTT/ModelContainer.h`, `research/model_container.py`):

```
header 16 B : "TSTT" | version u16 | section_count u16 | total_size u32 | crc32 u32
table       : section_count × 48 B: name[32] | dtype u8 | pad[3] | offset u32 | size u32 | count u32
data        : each section 16-byte aligned; the CRC covers bytes [16, total_size)
```

dtypes: 0 f32, 1 i8, 2 u8, 3 i16, 4 i32, 5 u32, 6 raw bytes.

| Sections | Content | Used by |
|---|---|---|
| `model.config` | u32 [version, n_mels, conv_channels, conv_kernel, conv_stride, gru_layers, hidden, n_classes] | `AcousticModel` |
| `conv.weight[.q/.scale]`, `gru{l}.weight_ih/hh[.q/.scale]`, `*.bias`, `out.*` | Weights: float32, and/or int8 with one f32 scale per row | `AcousticModel` |
| `features.mean`, `features.inv_std` | Feature normalization (80 values each) | `FeatureExtractor` |
| `lex.*` | Phoneme trie, word table and pronunciations (see `Lexicon.h`) | `Lexicon` |
| `lm.*` | Bigram LM: int16 natural-log values × 1000 | `LanguageModel` |
| `grammar.<name>` | Optional grammar text | `Grammar::compile(mc, name, ...)` |
| `info` | Free text / JSON | — |

A device that runs only one stage needs only that stage's sections. The
containers can be built separately and merged with `research/pack.py`.

## Loading

```cpp
ModelContainer model;
// 1. zero copy from a PROGMEM array or memory-mapped flash (4-byte aligned)
model.begin(tinystt_data, tinystt_data_len);
// 2. from a File / Stream into RAM (PSRAM on ESP32 and RP2350)
File f = LittleFS.open("/tinystt.tstt", "r");
model.begin(f, f.size());
// 3. ESP32: memory-mapped from the flash data partition "tinystt"
ModelPartition partition;
partition.begin(model, "tinystt");
```

To compile a model into the program, create its header (the library does not
ship them, they are large): `python research/export_header.py` turns every
model in `data/` into `src/TinySTT/data/<name>_model_data.h`
(`tinystt_model_data.h`, `tinystt_large_model_data.h`; one container with
explicit arguments). Then
`#include "TinySTT/data/tinystt_model_data.h"` and
`model.begin(tinystt_model_data, tinystt_model_data_len)` (see
[`model-headers.md`](model-headers.md)). It adds 3.1 MB to
the program (on ESP32 it is used in place from flash; choose a partition
scheme with a large app partition). Generated headers are ignored by git.

`ModelDownloader` (`TinySTT/ModelDownloader.h`) fetches a container with an
audio-tools URL stream (e.g. `URLStream`, which also logs into Wi-Fi) into a
file on SD, SD_MMC or LittleFS, once: a valid file that is already there is
kept, and an interrupted download never replaces it.

```cpp
URLStream url("ssid", "password");
ModelDownloader downloader(url);
downloader.download(SD_MMC, "/tinystt_large.tstt", "https://.../tinystt_large.tstt");
File f = SD_MMC.open("/tinystt_large.tstt", "r");
model.begin(f, f.size());
```

`begin()` checks magic, version, sizes and the CRC. Pass `verifyCrc = false`
for trusted memory-mapped flash to skip the scan.

## Sizes

| Part | Size |
|---|---|
| Acoustic model (production shape, int8) | ≈ 0.95 MB (float32 ≈ 3.6 MB) |
| Lexicon | 1.42 MB for 25,814 words, 2.6 MB for 50,000 words |
| Bigram LM | 0.78 MB from the training transcripts, 6.0 MB from the LibriSpeech LM corpus (`3-gram.pruned.3e-7`) |

The trained English model ships as `data/tinystt.tstt` (3.1 MB: acoustic
model 0.93 MB, 25,814-word lexicon 1.42 MB, bigram LM 0.78 MB) and as
`data/tinystt_large.tstt` (9.5 MB: the same acoustic model, 50,000-word
lexicon, LibriSpeech bigram LM); see [`../data/README.md`](../data/README.md). `test/data/placeholder_model.tstt`
has random weights, and `test/data/toy_model.tstt` is a toy task, both for
the tests. See [`training.md`](training.md).
