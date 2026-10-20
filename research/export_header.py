"""Writes a model container as a C/C++ header with a PROGMEM byte array.

    python research/export_header.py data/tinystt.tstt src/TinySTT/data/tinystt_model_data.h

The array is 16 byte aligned (the SIMD kernels read the int8 weights in place)
and placed with TINYSTT_PROGMEM (see src/TinySTT/DataAttr.h). Use it with
    model.begin(tinystt_model_data, tinystt_model_data_len);
On ESP32 the array stays in flash (memory mapped), no RAM copy.
"""
import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("container")
    ap.add_argument("header")
    ap.add_argument("--name", default=None, help="array name (default: from the header file name)")
    args = ap.parse_args()
    data = open(args.container, "rb").read()
    name = args.name or re.sub(r"\W", "_", os.path.splitext(os.path.basename(args.header))[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.header)), exist_ok=True)
    with open(args.header, "w") as f:
        f.write(f"// {name}: generated from {os.path.basename(args.container)} ({len(data)} bytes) by\n")
        f.write("// research/export_header.py; do not edit. See docs/model-data.md.\n")
        f.write("#pragma once\n#include <stddef.h>\n#include <stdint.h>\n#include \"TinySTT/DataAttr.h\"\n\n")
        f.write(f"alignas(16) static const uint8_t {name}[] TINYSTT_PROGMEM = {{\n")
        for i in range(0, len(data), 40):
            f.write(",".join(str(b) for b in data[i:i + 40]) + ",\n")
        f.write("};\n")
        f.write(f"static const size_t {name}_len = {len(data)};\n")
    print(f"wrote {args.header}: {name}[{len(data)}], {os.path.getsize(args.header)} bytes")


if __name__ == "__main__":
    main()
