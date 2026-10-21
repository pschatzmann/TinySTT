# Embedding a model (model compiled into the program)

A sketch can compile a model container into the program instead of loading it
from a file system, a flash partition or the network. The model is then used
in place from flash: no load step and no RAM copy.

The models in `data/` are part of the library in this form, in
`src/TinySTT/data/`:

| Model | Files | Array |
|---|---|---|
| `data/tinystt.tstt` (3.1 MB) | `tinystt_model.h`, `tinystt_model.S`, `tinystt_model.S.bin` | `tinystt_model`, `tinystt_model_size()` |
| `data/tinystt_large.tstt` (9.5 MB) | `tinystt_large_model.h`, `tinystt_large_model.S`, `tinystt_large_model.S.bin` | `tinystt_large_model`, `tinystt_large_model_size()` |

Nothing needs to be generated to use them. `python research/embed_model.py`
creates these files for every model in `data/` (run it after a model changes)
or, with arguments, for other containers (`python research/embed_model.py my.tstt`).

## Using a model

```cpp
#include "TinySTT/data/tinystt_model.h"

ModelContainer model;
model.begin(tinystt_model, tinystt_model_size());
```

[`examples/stt_waveshare_p4`](../examples/stt_waveshare_p4) uses the large
model this way.

## How it works

The `.S` file uses the assembler directive `.incbin`, which copies the
`.S.bin` file into the read-only data section (flash) when the sketch is
compiled, 16 byte aligned for the SIMD kernels:

```asm
  .macro tinystt_incbin path
  .incbin "\path\().bin"
  .endm
  .section .rodata.tinystt_model, "a"
  .balign 16
  .global tinystt_model
tinystt_model:
  tinystt_incbin __FILE__
  .global tinystt_model_end
tinystt_model_end:
```

- **Wherever the library is installed:** the preprocessor replaces `__FILE__` with the absolute path of the `.S` file, and the macro appends `.bin`. A relative path would not work: the assemblers do not search the folder of the source file, and the ESP32-S3 one does not even search the include paths.
- **Any GCC toolchain:** tested with ESP32-S3 (Xtensa), ESP32-P4 (RISC-V) and RP2350 (ARM), also with the library in another folder.
- **No cost for other sketches:** Arduino assembles the `.S` files for every sketch that uses TinySTT, but the linker drops the sections nobody references (`--gc-sections`): a sketch without a model does not grow.
- **Fast:** the assembler only copies bytes, so a sketch with the 9.5 MB model compiles in about a minute (a C array in a header took several minutes and 29 MB of source).

## Flash size

The model is part of the program, and every upload transfers it:

| Board | Program with the default model | Partition scheme |
|---|---:|---|
| ESP32-S3 (8/16 MB flash) | 3.6 MB | "RainMaker 8MB" (4 MB app) or a custom `partitions.csv` |
| ESP32-P4 (32 MB flash) | 3.6 MB; 10.3 MB with the large model | "32M Flash (13MB APP/6.75MB SPIFFS)" |
| RP2350 (16 MB flash) | 3.3 MB | the app part of the flash setting |

To flash a model only once instead of with every upload, write it to a data
partition and map it with `ModelPartition`, or download it with
`ModelDownloader` (see [`model-data.md`](model-data.md)).
