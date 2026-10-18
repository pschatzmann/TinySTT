"""Writes test/data/test_lexicon_lm.tstt: a small lexicon + bigram LM built from
an embedded mini dictionary and corpus (no download), used by test_lexicon,
test_lm and test_decoder. Also writes test_lm_reference.txt with probabilities
computed here, which the C++ test compares against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_lexicon  # noqa: E402
import build_lm  # noqa: E402
from model_container import ContainerWriter  # noqa: E402
from phonemes import phone_id  # noqa: E402

DICT = """
a AH
a(2) EY
an AE N
the DH AH
the(2) DH IY
hello HH AH L OW
hello(2) HH EH L OW
world W ER L D
turn T ER N
on AA N
off AO F
light L AY T
lights L AY T S
fan F AE N
kitchen K IH CH AH N
bedroom B EH D R UW M
to T UW
two T UW
too T UW
for F AO R
four F AO R
right R AY T
write R AY T
one W AH N
won W AH N
three TH R IY
go G OW
stop S T AA P
please P L IY Z
set S EH T
timer T AY M ER
minutes M IH N AH T S
what W AH T
time T AY M
is IH Z
it IH T
""".strip()

CORPUS = """
turn on the light
turn off the light
turn on the kitchen light
turn off the bedroom light
turn on the fan
turn off the fan
please turn on the light
please turn off the fan
set a timer for two minutes
set a timer for four minutes
set a timer for one minute
what time is it
hello world
write it right
go to the kitchen
go to the bedroom
stop the fan
turn on the lights
the light is on
the fan is off
it is two
one two three four
""".strip().splitlines()


def lexicon():
    lex = {}
    for line in DICT.splitlines():
        parts = line.split()
        word = parts[0].split("(")[0]
        lex.setdefault(word, []).append([phone_id(p) for p in parts[1:]])
    return lex


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test", "data")
    lex = lexicon()
    words, lex_sections = build_lexicon.build_sections(lex.keys(), lex)
    lm_sections = build_lm.build_sections(words, [s.split() for s in CORPUS], min_count=1)
    w = ContainerWriter()
    build_lexicon.add_sections(w, lex_sections)
    for name, value in lm_sections.items():
        w.add(name, value)
    w.add_string("info", "TinySTT test lexicon + LM (embedded mini corpus)")
    size = w.write(os.path.join(out_dir, "test_lexicon_lm.tstt"))
    # reference log probabilities (dequantized like the C++ reads them)
    ids = {wd: i for i, wd in enumerate(words)}
    V0 = len(words)
    BOS, EOS, UNK = V0, V0 + 1, V0 + 2
    scale = 1000.0
    uni, bo = lm_sections["lm.uni_logp"], lm_sections["lm.uni_backoff"]
    off, bw, bl = lm_sections["lm.bi_offsets"], lm_sections["lm.bi_words"], lm_sections["lm.bi_logp"]

    def logp(h, wd):
        for j in range(off[h], off[h + 1]):
            if bw[j] == wd:
                return bl[j] / scale
        return bo[h] / scale + uni[wd] / scale

    pairs = [("<s>", "turn"), ("turn", "on"), ("on", "the"), ("the", "light"), ("light", "</s>"),
             ("the", "kitchen"), ("hello", "world"), ("world", "stop"), ("timer", "for"),
             ("for", "two"), ("for", "too"), ("<s>", "minutes"), ("bedroom", "<unk>")]
    name = {"<s>": BOS, "</s>": EOS, "<unk>": UNK}
    with open(os.path.join(out_dir, "test_lm_reference.txt"), "w") as f:
        for h, wd in pairs:
            hi, wi = name.get(h, ids.get(h)), name.get(wd, ids.get(wd))
            f.write(f"{h} {wd} {logp(hi, wi):.6f}\n")
    print(f"wrote test_lexicon_lm.tstt ({size} bytes, {len(words)} words) and test_lm_reference.txt")


if __name__ == "__main__":
    main()
