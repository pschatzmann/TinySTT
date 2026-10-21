"""Writes a model container as a C/C++ header with a PROGMEM byte array.

The library does not ship these headers (they are large); create them when a
sketch wants the model compiled into the program:

    python research/export_header.py                           # every data/*.tstt
    python research/export_header.py my.tstt my_model_data.h   # one container

Without arguments, data/<name>.tstt becomes src/TinySTT/data/<name>_model_data.h
with the array <name>_model_data, e.g. tinystt_model_data and
tinystt_large_model_data.

The array is 16 byte aligned (the SIMD kernels read the int8 weights in place)
and placed with TINYSTT_PROGMEM (see src/TinySTT/DataAttr.h). Use it with
    #include "TinySTT/data/tinystt_model_data.h"
    model.begin(tinystt_model_data, tinystt_model_data_len);
On ESP32 the array stays in flash (memory mapped), no RAM copy. The array
name is the header's file name.
"""
import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_header(container, header, name=None):
    data = open(container, "rb").read()
    name = name or re.sub(r"\W", "_", os.path.splitext(os.path.basename(header))[0])
    os.makedirs(os.path.dirname(os.path.abspath(header)), exist_ok=True)
    with open(header, "w") as f:
        f.write(f"// {name}: generated from {os.path.basename(container)} ({len(data)} bytes) by\n")
        f.write("// research/export_header.py; do not edit. See docs/model-data.md.\n")
        f.write("#pragma once\n#include <stddef.h>\n#include <stdint.h>\n#include \"TinySTT/DataAttr.h\"\n\n")
        f.write(f"alignas(16) static const uint8_t {name}[] TINYSTT_PROGMEM = {{\n")
        for i in range(0, len(data), 40):
            f.write(",".join(str(b) for b in data[i:i + 40]) + ",\n")
        f.write("};\n")
        f.write(f"static const size_t {name}_len = {len(data)};\n")
    print(f"wrote {header}: {name}[{len(data)}], {os.path.getsize(header)} bytes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container", nargs="?", help="model container (default: every data/*.tstt)")
    ap.add_argument("header", nargs="?", help="header to write (default: src/TinySTT/data/<name>_model_data.h)")
    ap.add_argument("--name", default=None, help="array name (default: from the header file name)")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "src", "TinySTT", "data")
    containers = [args.container] if args.container else sorted(glob.glob(os.path.join(ROOT, "data", "*.tstt")))
    if not containers:
        raise SystemExit("no .tstt files in data/")
    for container in containers:
        base = os.path.splitext(os.path.basename(container))[0]
        header = args.header or os.path.join(out_dir, base + "_model_data.h")
        write_header(container, header, args.name)


if __name__ == "__main__":
    main()
