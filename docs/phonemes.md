# Phonemes

The model outputs 41 classes: CTC blank, silence and the 39 ARPAbet phonemes of
the CMU Pronouncing Dictionary without stress. This is the same base set
TinyTTS uses (TinyTTS keeps stress as a separate tone id). The ids are stable:
they are the model outputs, the ids in `PhonemeFrame` and on the wire, and the
ids in the lexicon. Mirrored in `research/phonemes.py`.

| id | sym | id | sym | id | sym | id | sym | id | sym |
|---:|---|---:|---|---:|---|---:|---|---:|---|
| 0 | `<b>` (blank) | 9 | CH | 18 | IH | 27 | OY | 36 | V |
| 1 | sil | 10 | D | 19 | IY | 28 | P | 37 | W |
| 2 | AA | 11 | DH | 20 | JH | 29 | R | 38 | Y |
| 3 | AE | 12 | EH | 21 | K | 30 | S | 39 | Z |
| 4 | AH | 13 | ER | 22 | L | 31 | SH | 40 | ZH |
| 5 | AO | 14 | EY | 23 | M | 32 | T | | |
| 6 | AW | 15 | F | 24 | N | 33 | TH | | |
| 7 | AY | 16 | G | 25 | NG | 34 | UH | | |
| 8 | B | 17 | HH | 26 | OW | 35 | UW | | |

`Phonemes::id("AH1")` ignores the stress digit. Training uses no silence
targets yet: the decoder treats `sil` like blank.
