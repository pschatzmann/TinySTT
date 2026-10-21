# Model headers (model compiled into the program)

A sketch can compile a model container into the program as a PROGMEM byte
array instead of loading it from a file system or a flash partition. The
library does not ship these headers because they are large (9.5 MB of source
for the 3.1 MB model); create them when you need them:

```bash
python research/export_header.py
```

This turns every model in `data/` into a header in `src/TinySTT/data/`:

| Model | Header | Array |
|---|---|---|
| `data/tinystt.tstt` (3.1 MB) | `tinystt_model_data.h` | `tinystt_model_data`, `tinystt_model_data_len` |
| `data/tinystt_large.tstt` (9.5 MB) | `tinystt_large_model_data.h` | `tinystt_large_model_data`, `tinystt_large_model_data_len` |

`python research/export_header.py my.tstt my_model_data.h` converts a single
container. The generated headers are ignored by git.

## Using a header

```cpp
#include "TinySTT/data/tinystt_model_data.h"

ModelContainer model;
model.begin(tinystt_model_data, tinystt_model_data_len);
```

The array is 16 byte aligned (the SIMD kernels read the int8 weights in
place) and marked with `TINYSTT_PROGMEM` (see `src/TinySTT/DataAttr.h`). On
ESP32 it stays in flash and is used in place, without a RAM copy. The model
adds its size to the program: choose a partition scheme with a large app
partition, e.g. "32M Flash (13MB APP/6.75MB SPIFFS)" on the ESP32-P4
(see [`examples/stt_waveshare_p4`](../examples/stt_waveshare_p4)).

For the other ways to load a model (file system, flash partition, download)
see [`model-data.md`](model-data.md).
