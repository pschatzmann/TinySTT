"""Downloads LibriSpeech subsets (OpenSLR 12) and writes manifests.

    python research/prepare_librispeech.py --root /data/librispeech \
        --subsets train-clean-100 dev-clean test-clean

Manifest: one line per utterance "path<TAB>TRANSCRIPT" (path to the .flac).
Utterances with words missing from CMUdict are kept here and filtered by
train.py, so the manifest stays independent of the lexicon.
"""
import argparse
import os
import sys
import tarfile
import time
import urllib.request

# OpenSLR and its mirrors; tried in order
URLS = ["https://openslr.elda.org/resources/12/{}.tar.gz",
        "https://www.openslr.org/resources/12/{}.tar.gz",
        "https://openslr.magicdatatech.com/resources/12/{}.tar.gz"]


def log(msg):
    print(msg, flush=True)


def fetch(url, path):
    """Downloads with a progress line every 5 % (or 30 s)."""
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length", 0))
        done, t0, last_pct, last_t = 0, time.time(), -5, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            pct = 100 * done / total if total else 0
            now = time.time()
            if pct >= last_pct + 5 or now - last_t > 30:
                speed = done / max(now - t0, 1e-3) / 1e6
                eta = (total - done) / (speed * 1e6) / 60 if total and speed > 0 else 0
                log(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f} %), {speed:.1f} MB/s, ETA {eta:.0f} min")
                last_pct, last_t = pct, now
    os.replace(tmp, path)


def download(root, subset, delete_archive=False):
    target = os.path.join(root, "LibriSpeech", subset)
    if os.path.isdir(target):
        return target
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, subset + ".tar.gz")
    if not os.path.exists(archive):
        for url in URLS:
            log(f"downloading {url.format(subset)}")
            try:
                fetch(url.format(subset), archive)
                break
            except Exception as e:  # try the next mirror
                log(f"  failed: {e}")
        else:
            sys.exit(f"could not download {subset}")
    log(f"extracting {archive}")
    with tarfile.open(archive) as tar:
        for n, member in enumerate(tar, 1):
            tar.extract(member, root, filter="data")
            if n % 10000 == 0:
                log(f"  {n} files")
    if delete_archive:
        os.remove(archive)
    return target


def manifest(subset_dir, out_path):
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for dirpath, _, files in sorted(os.walk(subset_dir)):
            for name in sorted(files):
                if not name.endswith(".trans.txt"):
                    continue
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    for line in f:
                        utt, text = line.strip().split(" ", 1)
                        out.write(f"{os.path.join(dirpath, utt + '.flac')}\t{text}\n")
                        n += 1
    log(f"{out_path}: {n} utterances")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--subsets", nargs="+", default=["train-clean-100", "dev-clean", "test-clean"])
    ap.add_argument("--out", default=None, help="manifest directory (default: root)")
    ap.add_argument("--delete-archive", action="store_true", help="remove each .tar.gz after extraction")
    args = ap.parse_args()
    out = args.out or args.root
    os.makedirs(out, exist_ok=True)
    for s in args.subsets:
        manifest(download(args.root, s, args.delete_archive), os.path.join(out, s + ".tsv"))


if __name__ == "__main__":
    main()
