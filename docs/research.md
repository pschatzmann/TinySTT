# Research tooling (`research/`)

| File | Purpose |
|---|---|
| `features.py` | numpy copy of the device feature pipeline, including audio-tools' details (the Hann window is evaluated in float32, the windowed sample is truncated to int16, the Nyquist bin is dropped) |
| `model_numpy.py` | numpy reference of `AcousticModel` (float32 and int8) and the int8 quantization |
| `model_container.py` | Container reader/writer |
| `phonemes.py` | Phoneme table, CMUdict loader, transcript → phonemes |
| `prepare_librispeech.py` | Download and manifests |
| `train.py` | Training, QAT, export |
| `eval.py` | Greedy PER of a container |
| `build_lexicon.py`, `build_lm.py`, `pack.py` | Lexicon, LM (also from gzip text with `--max-lines`), merging |
| `export_header.py` | A container as C/C++ header with a 16-byte aligned PROGMEM array (`src/TinySTT/data/tinystt_model_data.h`) |
| `arpa_to_lm.py` | Bigram LM from the unigrams and bigrams of an ARPA model (e.g. the LibriSpeech models) |
| `eval_wer.py` | Word error rate of the C++ pipeline (`tinystt` CLI) on a manifest |
| `rescore_nbest.py` | Word error rate with N-best rescoring by a larger ARPA model (second-pass experiment) |
| `make_*.py` | Test fixtures (see [`training.md`](training.md)), the benchmark container (`make_bench_model.py`) and the self-test data (`make_selftest_data.py`) |
| `kaggle_train.ipynb`, `Dockerfile` | Training environments |

Run Python with `-I` when you work on downloaded data (isolated mode).
