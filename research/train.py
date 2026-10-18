"""Trains TinySTT's streaming CTC phoneme model and exports the model container.

The PyTorch model is the exact counterpart of src/TinySTT/AcousticModel.h:
causal Conv1d (kernel K, stride S, K-1 left padding) + ReLU -> GRU x L ->
Linear -> log-softmax, trained with CTC on ARPAbet phonemes (CMUdict). Inputs
are the device features from research/features.py, normalized with global
statistics that are stored in the container ("features.mean/inv_std").

The last --qat-epochs use quantization-aware training: the forward pass uses
per-row int8 fake-quantized weights (straight-through estimator), so the int8
export loses little accuracy.

    python research/train.py --train data/train-clean-100.tsv --dev data/dev-clean.tsv \
        --out runs/base --epochs 30 --qat-epochs 3
    python research/train.py --export runs/base/best.pt --container model.tstt

See docs/training.md (Kaggle notebook: research/kaggle_train.ipynb).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat  # noqa: E402
import model_numpy as mn  # noqa: E402
from model_container import ContainerWriter  # noqa: E402
from phonemes import PHONEMES, load_cmudict, text_to_phonemes  # noqa: E402

DEFAULT_CONFIG = dict(version=1, n_mels=80, conv_channels=256, conv_kernel=5, conv_stride=3,
                      gru_layers=2, hidden=256, n_classes=len(PHONEMES))


# ---------------------------------------------------------------- audio / data

def load_audio(path, sr=feat.SAMPLE_RATE):
    """Any soundfile-readable file -> int16 mono at 16 kHz (linear resampling)."""
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if rate != sr:
        x = resample(x, rate, sr)
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def resample(x, rate_in, rate_out):
    n_out = int(round(len(x) * rate_out / rate_in))
    t_out = np.arange(n_out) * (rate_in / rate_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


def read_manifest(path, lexicon, max_seconds=20.0):
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            audio, text = line.rstrip("\n").split("\t", 1)
            ph = text_to_phonemes(text, lexicon)
            if not ph:
                skipped += 1
                continue
            items.append((audio, text, ph))
    print(f"{path}: {len(items)} utterances ({skipped} skipped: words not in lexicon)", flush=True)
    return items


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, items, mean, inv_std, augment=False):
        self.items, self.mean, self.inv_std, self.augment = items, mean, inv_std, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, _, ph = self.items[idx]
        pcm = load_audio(path)
        if self.augment:
            pcm = augment_audio(pcm)
        x = feat.log_mel(pcm, mean=self.mean, inv_std=self.inv_std)
        if self.augment:
            x = spec_augment(x)
        return torch.from_numpy(x), torch.tensor(ph, dtype=torch.long)


def augment_audio(pcm):
    x = pcm.astype(np.float32)
    speed = random.choice([0.9, 1.0, 1.0, 1.1])
    if speed != 1.0:
        x = resample(x, feat.SAMPLE_RATE * speed, feat.SAMPLE_RATE)
    x *= 10 ** (random.uniform(-6, 6) / 20)  # gain
    if random.random() < 0.5:  # white noise at 10..40 dB SNR
        power = np.mean(x ** 2) + 1e-9
        snr = random.uniform(10, 40)
        x += np.random.normal(0, math.sqrt(power / 10 ** (snr / 10)), len(x))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def spec_augment(x, freq_masks=2, freq_width=10, time_masks=2, time_width=20):
    x = x.copy()
    T, Fb = x.shape
    for _ in range(freq_masks):
        w = random.randint(0, freq_width)
        f0 = random.randint(0, max(0, Fb - w))
        x[:, f0:f0 + w] = 0.0
    for _ in range(time_masks):
        w = random.randint(0, min(time_width, max(1, T // 5)))
        t0 = random.randint(0, max(0, T - w))
        x[t0:t0 + w, :] = 0.0
    return x


def collate(batch):
    xs, ys = zip(*batch)
    x_len = torch.tensor([len(x) for x in xs])
    y_len = torch.tensor([len(y) for y in ys])
    x = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    y = torch.cat(ys)
    return x, x_len, y, y_len


def length_batches(items, batch_frames, shuffle=True):
    """Buckets utterances of similar duration (by file size) into batches."""
    sizes = [os.path.getsize(p) for p, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: sizes[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        cur_max = max(cur_max, sizes[i])
        if cur and cur_max * (len(cur) + 1) > batch_frames:
            batches.append(cur)
            cur, cur_max = [], sizes[i]
        cur.append(i)
    if cur:
        batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


def feature_stats(items, n=500):
    """Global per-bin mean / inverse std of the raw log-mel features."""
    rng = random.Random(0)
    sample = rng.sample(items, min(n, len(items)))
    s, s2, count = np.zeros(feat.MEL_BINS), np.zeros(feat.MEL_BINS), 0
    for path, _, _ in sample:
        x = feat.log_mel(load_audio(path)).astype(np.float64)
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
        count += len(x)
    mean = s / count
    std = np.sqrt(np.maximum(s2 / count - mean ** 2, 1e-6))
    return mean.astype(np.float32), (1.0 / std).astype(np.float32)


# ---------------------------------------------------------------- model

class AcousticModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(cfg["n_mels"], cfg["conv_channels"], cfg["conv_kernel"], stride=cfg["conv_stride"])
        self.gru = nn.GRU(cfg["conv_channels"], cfg["hidden"], cfg["gru_layers"], batch_first=True)
        self.out = nn.Linear(cfg["hidden"], cfg["n_classes"])

    def forward(self, x):
        """x [B, T, n_mels] -> log-probs [B, ceil(T/S), n_classes]"""
        x = F.pad(x.transpose(1, 2), (self.cfg["conv_kernel"] - 1, 0))
        x = F.relu(self.conv(x)).transpose(1, 2)
        x, _ = self.gru(x)
        return F.log_softmax(self.out(x), dim=-1)

    def output_lengths(self, x_len):
        return (x_len + self.cfg["conv_stride"] - 1) // self.cfg["conv_stride"]


def fake_quant_rows(w):
    """Per-row symmetric int8 fake quantization with a straight-through estimator."""
    w2 = w.reshape(w.shape[0], -1)
    scale = (w2.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(w2 / scale), -127, 127) * scale
    return (w2 + (q - w2).detach()).reshape(w.shape)


def qat_forward(model, x):
    params = {n: (fake_quant_rows(p) if p.dim() >= 2 else p) for n, p in model.named_parameters()}
    return torch.func.functional_call(model, params, (x,))


def numpy_weights(model):
    """state_dict -> names used by the container / model_numpy."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    w = {"conv.weight": sd["conv.weight"], "conv.bias": sd["conv.bias"],
         "out.weight": sd["out.weight"], "out.bias": sd["out.bias"]}
    for l in range(model.cfg["gru_layers"]):
        for kind in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
            w[f"gru{l}.{kind}"] = sd[f"gru.{kind}_l{l}"]
    return w


# ---------------------------------------------------------------- metrics

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def greedy_decode(logp, length):
    ids = logp[:length].argmax(-1).tolist()
    out, last = [], None
    for i in ids:
        if i != last and i != 0:
            out.append(i)
        last = i
    return out


@torch.no_grad()
def evaluate(model, loader, device, quantized=False):
    model.eval()
    errors, total, loss_sum, n = 0, 0, 0.0, 0
    for x, x_len, y, y_len in loader:
        x = x.to(device)
        logp = qat_forward(model, x) if quantized else model(x)
        out_len = model.output_lengths(x_len)
        loss_sum += F.ctc_loss(logp.transpose(0, 1).float(), y, out_len, y_len, blank=0,
                               zero_infinity=True).item()
        n += 1
        offset = 0
        for b in range(x.shape[0]):
            ref = y[offset:offset + y_len[b]].tolist()
            offset += y_len[b]
            errors += edit_distance(greedy_decode(logp[b].cpu(), out_len[b]), ref)
            total += len(ref)
    model.train()
    return errors / max(total, 1), loss_sum / max(n, 1)


# ---------------------------------------------------------------- export

def export(checkpoint, container_path, float32=False, int8=True):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = AcousticModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    weights = numpy_weights(model)
    # torch and the numpy reference (= the C++ semantics) must agree
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (37, cfg["n_mels"])).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)[None])[0].numpy()
    diff = np.abs(ref - mn.forward(cfg, weights, x)).max()
    print(f"torch vs numpy reference: max diff {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("exported model does not match the reference implementation")
    w = ContainerWriter()
    mn.add_model(w, cfg, weights, int8=int8, float32=float32, mean=ck["mean"], inv_std=ck["inv_std"])
    w.add_string("info", json.dumps({"checkpoint": os.path.basename(checkpoint),
                                     "dev_per": ck.get("dev_per"), "epoch": ck.get("epoch")}))
    size = w.write(container_path)
    print(f"wrote {container_path} ({size} bytes)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", help="training manifest(s)")
    ap.add_argument("--dev", help="dev manifest")
    ap.add_argument("--out", default="runs/default")
    ap.add_argument("--config", default=None, help="JSON with model shape overrides")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--batch-bytes", type=int, default=12_000_000, help="audio bytes per batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lexicon", default=None, help="cmudict.dict (downloaded if missing)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-items", type=int, default=0, help="limit for smoke tests")
    ap.add_argument("--no-augment", action="store_true", help="disable audio/SpecAugment augmentation")
    ap.add_argument("--log-every", type=int, default=200, help="progress line every N batches")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop training after this many hours (then QAT/export still run)")
    ap.add_argument("--export", default=None, help="checkpoint to export (skips training)")
    ap.add_argument("--container", default="model.tstt")
    ap.add_argument("--export-float32", action="store_true", help="also store float32 weights")
    args = ap.parse_args()

    if args.export:
        export(args.export, args.container, float32=args.export_float32)
        return

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    os.makedirs(args.out, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lexicon = load_cmudict(args.lexicon)
    train_items = [it for m in args.train for it in read_manifest(m, lexicon)]
    dev_items = read_manifest(args.dev, lexicon) if args.dev else train_items[:200]
    if args.max_items:
        train_items, dev_items = train_items[:args.max_items], dev_items[:args.max_items]

    stats_path = os.path.join(args.out, "feature_stats.npz")
    if os.path.exists(stats_path):
        st = np.load(stats_path)
        mean, inv_std = st["mean"], st["inv_std"]
    else:
        print("computing feature statistics (500 utterances) ...", flush=True)
        mean, inv_std = feature_stats(train_items)
        np.savez(stats_path, mean=mean, inv_std=inv_std)

    train_ds = SpeechDataset(train_items, mean, inv_std, augment=not args.no_augment)
    dev_ds = SpeechDataset(dev_items, mean, inv_std)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_sampler=length_batches(dev_items, args.batch_bytes, False),
                                             collate_fn=collate, num_workers=args.workers)

    model = AcousticModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = len(length_batches(train_items, args.batch_bytes, False))
    print(f"model: {sum(p.numel() for p in model.parameters())} params, config {cfg}, device {device}, "
          f"{steps_per_epoch} batches/epoch", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    start_epoch, best = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck.get("dev_per", best)

    t_start = time.time()
    epoch_times = []
    end_epoch = args.epochs
    use_sched = True
    epoch = start_epoch
    while epoch < end_epoch:
        qat = epoch >= end_epoch - args.qat_epochs
        if args.max_hours and epoch_times and not qat:
            # leave room for the QAT epochs, evaluation and the export
            avg = sum(epoch_times) / len(epoch_times)
            if time.time() - t_start + avg * (args.qat_epochs + 1) > args.max_hours * 3600:
                print(f"time limit: QAT from epoch {epoch} with a low constant learning rate", flush=True)
                end_epoch = epoch + args.qat_epochs
                qat = True
                use_sched = False  # the OneCycle schedule was cut short
                for g in opt.param_groups:
                    g["lr"] = args.lr * 0.05
        loader = torch.utils.data.DataLoader(train_ds, batch_sampler=length_batches(train_items, args.batch_bytes),
                                             collate_fn=collate, num_workers=args.workers)
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, x_len, y, y_len in loader:
            x = x.to(device)
            logp = qat_forward(model, x) if qat else model(x)
            loss = F.ctc_loss(logp.transpose(0, 1).float(), y, model.output_lengths(x_len), y_len,
                              blank=0, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if use_sched:
                sched.step()
            loss_sum += loss.item()
            n += 1
            if args.log_every and n % args.log_every == 0:
                el = time.time() - t0
                print(f"  epoch {epoch} batch {n}/{steps_per_epoch}: loss {loss_sum / n:.3f}, "
                      f"{el / 60:.1f} min, epoch ETA {(steps_per_epoch - n) * el / n / 60:.1f} min, "
                      f"total {(time.time() - t_start) / 3600:.2f} h", flush=True)
        per, dev_loss = evaluate(model, dev_loader, device, quantized=qat)
        epoch_times.append(time.time() - t0)
        print(f"epoch {epoch}{' (qat)' if qat else ''}: train loss {loss_sum / max(n, 1):.3f}, "
              f"dev loss {dev_loss:.3f}, dev PER {per:.3f}, {time.time() - t0:.0f} s", flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "config": cfg, "epoch": epoch, "dev_per": per, "mean": mean, "inv_std": inv_std, "qat": qat}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        # during QAT the quantized PER is what the device will see
        if per < best or (qat and epoch == end_epoch - 1):
            best = min(best, per)
            torch.save(ck, os.path.join(args.out, "best.pt"))
        epoch += 1
    export(os.path.join(args.out, "best.pt"), os.path.join(args.out, "model.tstt"))


if __name__ == "__main__":
    main()
