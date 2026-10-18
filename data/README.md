# Model data

| File | Size | Content |
|---|---:|---|
| `tinystt.tstt` | 3.1 MB | English model container: acoustic model (0.9 M parameters, int8, 0.93 MB), lexicon (25,814 CMUdict words, 1.42 MB) and bigram language model (0.78 MB) |

The acoustic model was trained on LibriSpeech train-clean-100 (CC BY 4.0)
with `research/kaggle_train.ipynb`. Results on LibriSpeech test-clean (error
rate = substituted, deleted and inserted units / reference length):
- greedy phoneme error rate 14.8 % (500 utterances, int8);
- word error rate of the complete C++ pipeline 36.0 % (beam 16) and 32.7 % (beam 32), all 2,620 utterances.

Load it from a file system or embed it as an array; see
[`docs/model-data.md`](../docs/model-data.md). For a split setup, the sender
only needs the acoustic model sections and the receiver only the lexicon and
LM sections (`research/pack.py` builds such containers).
