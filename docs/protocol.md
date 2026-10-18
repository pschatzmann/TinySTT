# Phoneme frames and wire protocol

## `PhonemeFrame` (queue message, 16 bytes)

| Field | Type | Meaning |
|---|---|---|
| `index` | u32 | Index of the (first) acoustic frame (30 ms units) |
| `type` | u8 | 0 `Frame`, 1 `BlankRun`, 2 `EndOfUtterance`, 3 `Reset` |
| `k` | u8 | Valid entries in `id`/`logprob` (≤ 4) |
| `run` | u16 | Acoustic frames represented (`BlankRun` length, 1 for `Frame`) |
| `id[4]` | u8 | Phoneme class ids, best first (see [`phonemes.md`](phonemes.md)) |
| `logprob[4]` | i8 | Natural-log probability × 16, clamped to [−128, 0] (ln p ∈ [−8, 0]) |

Phonemes outside the top-k are scored one unit (ln) below the k-th entry.

## Wire format (`PhonemeProtocol`, used by `QueueToStream` / `StreamToQueue`)

All fields are little endian:

```
0xA5 | seq u8 | type u8 | index u32 | run u16 | k u8 | k × (id u8, logprob i8) | crc8
```

- The size is 11 + 2k bytes: 19 for a full frame, 11 for a `BlankRun` or `EndOfUtterance`.
- The CRC-8 (poly 0x07) covers everything after the sync byte.
- `seq` increments per frame. The receiver counts gaps (`lostFrames()`).

Robustness:
- On a CRC error or an impossible header, the parser drops one byte and searches for the next sync byte.
- A valid frame that starts inside a corrupted one is still found.
- Frames can be split arbitrarily across reads.

`QueueToStream::setTextMode(true)` (CLI: `--text-phonemes`) and
`PhonemeOutput` in `Frames` mode print readable lines for debugging instead:

```
#0 S:-0.19 T:-2.19 G:-2.56 P:-6.69
#2 <blank x18>
#57 <eou>
```

`PhonemeOutput` in its default `Phonemes` mode prints only the most likely
phoneme sequence of each utterance, e.g. `S T AA P DH AH F AE N`. It drops
blanks and silence and collapses repeats.

Bandwidth: at most 33 frames/s × 19 bytes ≈ 630 B/s. With blank skipping it
is typically 5–10 × less, so a 115200 baud UART or ESP-NOW is plenty.
