# Training

The acoustic model is trained with PyTorch in `research/`. The lexicon and LM
are built from CMUdict and transcripts. Everything ends up in one container.

## On Kaggle (recommended, free GPU)

Open `research/kaggle_train.ipynb` in Kaggle and enable GPU and Internet in
the notebook settings, then run all cells. The notebook:

1. downloads LibriSpeech train-clean-100, dev-clean and test-clean and writes manifests (`prepare_librispeech.py`);
2. trains the model with CTC (30 epochs, the last 3 quantization-aware) and exports `model.tstt` (`train.py`);
3. builds a lexicon of up to 40 k words (25,814 for train-clean-100: the transcript words found in CMUdict) and a bigram LM from the transcripts and packs everything into `tinystt.tstt` (`build_lexicon.py`, `build_lm.py`, `pack.py`);
4. reports the greedy phoneme error rate on test-clean (`eval.py`).

## Results of the shipped model

PER = phoneme error rate, WER = word error rate: (substitutions + deletions +
insertions) / length of the reference (see
[`performance.md`](performance.md#accuracy-and-decoder-tuning)).

Trained on a Kaggle T4 in 5.3 hours (30 epochs, the last 3 quantization-aware):
- dev PER 17.3 %;
- test-clean greedy PER 14.8 %;
- WER of the complete pipeline 36.0 % at beam 16 (32.7 % at beam 32);
- TinyTTS → TinySTT round trip (`test_roundtrip`): WER 33 %, 2 of 4 sentences exact.

To improve it:
- **Use the skipped data:** 10,841 of 28,539 train-clean-100 utterances were skipped because a word is missing from CMUdict. Letter-to-sound pronunciations would recover them.
- **More epochs or more data** (train-clean-360): the dev PER was still falling.
- **A larger language model** from the LibriSpeech LM corpus (800 M words): without an LM the WER is 60–80 %, so the LM matters a lot.

Use `research/eval_wer.py` to measure the WER of the C++ pipeline on a manifest.

## Locally / Docker

```bash
pip install -r research/requirements.txt         # or: docker build -t tinystt-research research
python research/prepare_librispeech.py --root data --subsets train-clean-100 dev-clean
python research/train.py --train data/train-clean-100.tsv --dev data/dev-clean.tsv --out runs/base
python research/build_lexicon.py --vocab-from data/train-clean-100.tsv --max-words 40000 --out lexicon.tstt
python research/build_lm.py --text-from data/train-clean-100.tsv --lexicon lexicon.tstt --out lm.tstt
python research/pack.py runs/base/model.tstt lexicon.tstt lm.tstt --out tinystt.tstt
```

### Training details

| Aspect | Setting |
|---|---|
| Features | `research/features.py`: a bit-faithful copy of the device `FeatureExtractor` (verified by `test_features`). Global mean/std normalization is stored in the container. |
| Augmentation | Speed 0.9/1.1, gain, white noise (10–40 dB SNR), SpecAugment. `--no-augment` turns it off. |
| Optimizer | AdamW with OneCycle LR. Batches are bucketed by duration (`--batch-bytes`). |
| QAT | The last `--qat-epochs` run with per-row int8 fake-quantized weights (straight-through estimator). |
| Export | Checks that PyTorch and the numpy reference of the C++ model agree, then writes int8 weights (`--export-float32` also stores float32). |
| Model shape | `--config shape.json`, e.g. `{"hidden": 192}`. Keep widths multiples of 16, and `test_budget` limits in mind. |

### Your own vocabulary or language

- **Vocabulary:** `build_lexicon.py --words my_words.txt` (the words must be in CMUdict), or add words at runtime with `addWord()` / `addHotword()`.
- **LM:** `build_lm.py --text-from my_sentences.txt` (plain text, one sentence per line).
- **Commands:** use a grammar instead of an LM (`setGrammar()`).

## Test fixtures

| Script | Output |
|---|---|
| `research/make_placeholder_model.py` | Random-weight model plus numpy goldens (`test_model`) |
| `research/features.py --golden` | Feature golden (`test_features`) |
| `research/make_test_data.py` | Mini lexicon + LM from an embedded corpus (`test_lexicon`, `test_decoder`) |
| `research/make_toy_task.py` | Trains a small model on a toy "phoneme tone" task: real end-to-end tests without a speech corpus (`test_pipeline`). Needs PyTorch, about 30 minutes on CPU. |
