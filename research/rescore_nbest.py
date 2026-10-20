"""Word error rate with N-best rescoring by a larger ARPA language model.

The tinystt CLI decodes every utterance (first pass with the container's
bigram) and prints its N best hypotheses with their total scores. Each
hypothesis is then rescored: the first-pass LM score is replaced by the
score of the ARPA model (e.g. a trigram), which is what an SD card based
second pass on the device would do.

    python research/rescore_nbest.py --cli build/desktop/tinystt --container tinystt.tstt \\
        --manifest test-clean.tsv --arpa 3-gram.pruned.1e-7.arpa.gz --nbest 16

score' = score + lm_weight * (ln P_arpa(words) - ln P_bigram(words)) + bonus * #words
P_bigram uses the unigrams and bigrams of the same ARPA file (the first-pass
LM built from it by arpa_to_lm.py), so only the higher orders are added.
"""
import argparse
import gzip
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_wer import edit_distance, load_pcm16k  # noqa: E402

LN10 = math.log(10.0)


class Arpa:
    def __init__(self, path):
        self.p = [None, {}, {}, {}]
        self.bo = [None, {}, {}, {}]
        order = 0
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("\\"):
                    order = int(line[1]) if line[1].isdigit() else 0
                    continue
                if order == 0:
                    continue
                parts = line.split()
                key = tuple(w.lower() for w in parts[1:1 + order])
                self.p[order][key] = float(parts[0])
                if len(parts) > 1 + order:
                    self.bo[order][key] = float(parts[1 + order])
        self.order = max(o for o in (1, 2, 3) if self.p[o])

    def logp(self, hist, w, order):
        """log10 p(w | hist) with backoff, using at most `order` words."""
        hist = tuple(hist[-(order - 1):]) if order > 1 else ()
        while True:
            key = hist + (w,)
            p = self.p[len(key)].get(key)
            if p is not None:
                return p
            if not hist:
                return self.p[1].get(("<unk>",), -7.0)
            back = self.bo[len(hist)].get(hist, 0.0)
            hist = hist[1:]
            return back + self.logp(hist, w, len(hist) + 1)

    def sentence(self, words, order):
        hist, total = ["<s>"], 0.0
        for w in words + ["</s>"]:
            total += self.logp(hist, w if w == "</s>" or (w,) in self.p[1] else "<unk>", order)
            hist.append(w)
        return total * LN10


def nbest_lists(args, items):
    for path, ref in items:
        cmd = [args.cli, "--model", args.container, "--no-vad", "--nbest", str(args.nbest)] + args.args.split() + ["-"]
        out = subprocess.run(cmd, input=load_pcm16k(path), capture_output=True, check=True).stdout.decode()
        hyps = []
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                hyps.append((float(parts[1]), parts[2].split()))
        if not hyps:  # nothing recognized (or a single result printed as text)
            hyps = [(0.0, out.split())]
        yield ref.lower().split(), hyps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arpa", required=True)
    ap.add_argument("--nbest", type=int, default=16)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--args", default="", help="additional CLI arguments")
    ap.add_argument("--weights", default="0.5", help="comma separated LM weights to evaluate")
    ap.add_argument("--bonus", default="0", help="comma separated word bonuses to evaluate")
    args = ap.parse_args()
    items = [line.rstrip("\n").split("\t", 1) for line in open(args.manifest, encoding="utf-8") if "\t" in line]
    if args.max_items:
        items = items[: args.max_items]
    print(f"loading {args.arpa}", file=sys.stderr)
    lm = Arpa(args.arpa)
    data = []
    for ref, hyps in nbest_lists(args, items):
        data.append((ref, [(s, ws, lm.sentence(ws, lm.order) - lm.sentence(ws, 2)) for s, ws in hyps]))
    n_words = sum(len(r) for r, _ in data)
    first = sum(edit_distance(r, h[0][1]) for r, h in data)
    oracle = sum(min(edit_distance(r, ws) for _, ws, _ in h) for r, h in data)
    print(f"{len(data)} utterances: first pass WER {first / n_words:.3f}, N-best oracle {oracle / n_words:.3f}")
    for w in (float(x) for x in args.weights.split(",")):
        for b in (float(x) for x in args.bonus.split(",")):
            errors = 0
            for ref, hyps in data:
                best = max(hyps, key=lambda h: h[0] + w * h[2] + b * len(h[1]))
                errors += edit_distance(ref, best[1])
            print(f"rescored (order {lm.order}, lm weight {w}, bonus {b}): WER {errors / n_words:.3f}")


if __name__ == "__main__":
    main()
