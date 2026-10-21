# Training

The acoustic model is trained with PyTorch in `research/`. The lexicon and LM
are built from CMUdict and transcripts. Everything ends up in one container.

## On Kaggle (recommended, free GPU)

Open `research/kaggle_train.ipynb` in Kaggle and enable GPU and Internet in
the notebook settings, attach the dataset `victorling/librispeech-clean`
(LibriSpeech, extracted) and run all cells. The notebook:

1. links the attached LibriSpeech subsets (downloads missing ones) and writes manifests for train-clean-100, train-clean-360, dev-clean and test-clean (`prepare_librispeech.py`);
2. trains the model with CTC on 460 h of audio (7 epochs, the last one quantization-aware, about 80 minutes each on a T4) and exports `model.tstt` (`train.py`). The pronunciations come from the LibriSpeech lexicon (CMUdict + letter-to-sound for all LibriSpeech words), so only utterances with very rare words are skipped;
3. builds a lexicon of up to 40 k words (25,814 for train-clean-100: the transcript words found in CMUdict) and a bigram LM from the transcripts and packs everything into `tinystt.tstt` (`build_lexicon.py`, `build_lm.py`, `pack.py`);
4. reports the greedy phoneme error rate on test-clean (`eval.py`).

## Results of the shipped model

PER = phoneme error rate, WER = word error rate: (substitutions + deletions +
insertions) / length of the reference (see
[`performance.md`](performance.md#accuracy-and-decoder-tuning)).

Trained on a Kaggle T4 in 9.6 hours: train-clean-100 + train-clean-360 with
the LibriSpeech lexicon (122,904 utterances, 7 % skipped for very rare
words), 5 epochs plus 1 quantization-aware epoch (the time budget ended the
planned 7 epochs early; an epoch takes 85–135 minutes):

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 (QAT) |
|---|---:|---:|---:|---:|---:|---:|
| Dev PER | 28.5 % | 22.5 % | 19.8 % | 17.6 % | 16.1 % | 15.2 % |

- test-clean greedy PER 15.0 % (int8);
- WER of the complete pipeline 29.6 % at beam 32 with `data/tinystt.tstt`, 25.7 % at beam 64 with `data/tinystt_large.tstt`.

The previous model (train-clean-100 only, CMUdict pronunciations: 10,841 of
28,539 utterances skipped; 30 epochs, dev PER 17.3 %) gave 32.7 % and 29.8 %.
The dev PER was still falling: more epochs (a second Kaggle session with
`--resume`) are the next step.
- **A larger language model** from the LibriSpeech LM corpus (800 M words): without an LM the WER is 60–80 %, so the LM matters a lot.

Use `research/eval_wer.py` to measure the WER of the C++ pipeline on a manifest.

## A larger lexicon and language model

The public LibriSpeech language models (www.openslr.org/11, trained on 800 M
words) can replace the bigram from the training transcripts:

```bash
# 50 k most frequent words, pronunciations from the LibriSpeech lexicon
python research/build_lexicon.py --words words_50k.txt --max-words 50000 \
    --lexicon librispeech-lexicon.txt --out lexicon_50k.tstt
# unigrams + bigrams of the entropy-pruned ARPA model
python research/arpa_to_lm.py --arpa 3-gram.pruned.3e-7.arpa.gz --lexicon lexicon_50k.tstt --out lm_50k.tstt
python research/pack.py run/model.tstt lexicon_50k.tstt lm_50k.tstt --out tinystt_large.tstt
```

`words_50k.txt` lists the words by their unigram probability in the ARPA
model. The result is a 9.5 MB container (see
[`performance.md`](performance.md#larger-lexicon-and-language-model) for the
word error rates and device timings).

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
| Time budget | `--max-hours` is checked every 50 batches: an epoch that would leave too little time ends early, then the QAT epochs follow at a low learning rate, so that a run always finishes with an exported model (Kaggle stops sessions after 12 h). |
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
