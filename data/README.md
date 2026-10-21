# Model data

| File | Size | Content |
|---|---:|---|
| `tinystt.tstt` | 3.1 MB | English model container: acoustic model (0.9 M parameters, int8, 0.93 MB), lexicon (25,814 CMUdict words, 1.42 MB) and bigram language model from the training transcripts (0.78 MB) |
| `tinystt_large.tstt` | 9.5 MB | The same acoustic model with a 50,000-word lexicon (LibriSpeech lexicon, 2.6 MB) and a bigram language model from the LibriSpeech LM corpus (800 M words, 6.0 MB); use beam 64 |

The acoustic model was trained on LibriSpeech train-clean-100 and
train-clean-360 (460 h, CC BY 4.0) with `research/kaggle_train.ipynb`.
Results on all 2,620 LibriSpeech test-clean utterances (error rate =
substituted, deleted and inserted units / reference length):

| Container | Beam | Word error rate |
|---|---:|---:|
| `tinystt.tstt` | 32 (default) | 29.6 % |
| `tinystt_large.tstt` | 64 | **25.7 %** |

Greedy phoneme error rate: 15.0 % (500 utterances, int8). The large
container runs from flash on the ESP32-S3 (0.51 RTF at beam 64) and ESP32-P4
(0.16 RTF); it needs 9.5 MB flash or PSRAM.

Load it from a file system or embed it as an array; see
[`docs/model-data.md`](../docs/model-data.md). For a split setup, the sender
only needs the acoustic model sections and the receiver only the lexicon and
LM sections (`research/pack.py` builds such containers).
