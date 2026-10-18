"""Builds the toy end-to-end test fixture (requires PyTorch, runs on CPU in a few minutes):

Every phoneme is rendered as a distinct sine tone, words are phoneme tone
sequences separated by short pauses. A small model is trained on this with
train.py, so the C++ end-to-end tests run a *really trained* acoustic model
through the whole pipeline (VAD, features, model, queue, decoder) without
any speech corpus.

Writes to test/data:
  toy_model.tstt       acoustic model + lexicon + LM (one container)
  toy_utt{N}.wav       16 kHz test clips (one or two utterances each)
  toy_expected.txt     expected text per clip (utterances separated by '|')

    python research/make_toy_task.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "test", "data")

WORDS = {"turn": "T ER N", "on": "AA N", "off": "AO F", "the": "DH AH", "light": "L AY T",
         "fan": "F AE N", "go": "G OW", "stop": "S T AA P"}
PHONES = sorted({p for v in WORDS.values() for p in v.split()})
FREQ = {p: 250 * 1.2 ** i for i, p in enumerate(PHONES)}
SENTENCES = [["turn", "on", "the", "light"], ["turn", "off", "the", "light"], ["turn", "on", "the", "fan"],
             ["turn", "off", "the", "fan"], ["go"], ["stop"], ["stop", "the", "fan"], ["go", "on"]]
SR = 16000


def render(words, rng, lead=0.3, tail=0.3):
    audio = [np.zeros(int(lead * SR))]
    for w in words:
        for p in WORDS[w].split():
            t = np.arange(int(rng.uniform(0.08, 0.14) * SR)) / SR
            env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)  # 10 ms fades
            audio.append(rng.uniform(0.2, 0.4) * env * np.sin(2 * np.pi * FREQ[p] * t))
        audio.append(np.zeros(int(rng.uniform(0.08, 0.15) * SR)))
    audio.append(np.zeros(int(tail * SR)))
    x = np.concatenate(audio)
    return (x + rng.normal(0, 0.002, len(x))).astype(np.float32)


def main():
    rng = np.random.default_rng(5)
    work = tempfile.mkdtemp(prefix="tinystt_toy_")
    try:
        with open(os.path.join(work, "lex.dict"), "w") as f:
            for w, p in WORDS.items():
                f.write(f"{w} {p}\n")
        lines = []
        for n in range(480):
            words = SENTENCES[n % len(SENTENCES)]
            path = os.path.join(work, f"t{n}.wav")
            sf.write(path, render(words, rng), SR)
            lines.append(f"{path}\t{' '.join(words)}")
        open(os.path.join(work, "train.tsv"), "w").write("\n".join(lines[:440]) + "\n")
        open(os.path.join(work, "dev.tsv"), "w").write("\n".join(lines[440:]) + "\n")
        open(os.path.join(work, "cfg.json"), "w").write('{"conv_channels": 48, "hidden": 48}')
        py = sys.executable
        run = lambda *a: subprocess.run([py, *a], check=True, cwd=work)  # noqa: E731
        run(os.path.join(HERE, "train.py"), "--train", "train.tsv", "--dev", "dev.tsv", "--lexicon", "lex.dict",
            "--config", "cfg.json", "--epochs", "30", "--qat-epochs", "3", "--workers", "0",
            "--batch-bytes", "100000", "--lr", "5e-3", "--no-augment", "--out", "run")
        run(os.path.join(HERE, "build_lexicon.py"), "--vocab-from", "train.tsv", "--lexicon", "lex.dict",
            "--out", "lex.tstt")
        run(os.path.join(HERE, "build_lm.py"), "--text-from", "train.tsv", "--lexicon", "lex.tstt",
            "--min-count", "1", "--out", "lm.tstt")
        run(os.path.join(HERE, "pack.py"), "run/model.tstt", "lex.tstt", "lm.tstt",
            "--out", os.path.join(OUT, "toy_model.tstt"))
        # test clips: unseen renderings, some with two utterances separated by 1 s
        clips = [[["turn", "on", "the", "light"]], [["stop", "the", "fan"]], [["go"], ["turn", "off", "the", "fan"]]]
        test_rng = np.random.default_rng(99)
        with open(os.path.join(OUT, "toy_expected.txt"), "w") as f:
            for i, utts in enumerate(clips):
                parts = [render(u, test_rng, lead=0.5, tail=1.0) for u in utts]
                sf.write(os.path.join(OUT, f"toy_utt{i}.wav"), np.concatenate(parts), SR, subtype="PCM_16")
                f.write("|".join(" ".join(u) for u in utts) + "\n")
        print("toy fixture written to", os.path.abspath(OUT))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
