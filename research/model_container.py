"""Reader/writer for TinySTT's model container (src/TinySTT/ModelContainer.h).

Layout (little endian):
  header 16 B : b"TSTT" | version u16 | section_count u16 | total_size u32 | crc32 u32
  table       : section_count x 48 B: name[32] | dtype u8 | pad[3] | offset u32 | size u32 | count u32
  data        : each section 16 byte aligned; crc32 covers bytes [16, total_size)
"""
import struct
import zlib

import numpy as np

VERSION = 1
HEADER_SIZE = 16
ENTRY_SIZE = 48
NAME_SIZE = 32
ALIGN = 16

DTYPES = {  # code: numpy dtype
    0: np.dtype("<f4"),
    1: np.dtype("i1"),
    2: np.dtype("u1"),
    3: np.dtype("<i2"),
    4: np.dtype("<i4"),
    5: np.dtype("<u4"),
    6: np.dtype("u1"),  # raw bytes
}
BYTES = 6


def _align(v):
    return (v + ALIGN - 1) & ~(ALIGN - 1)


class ContainerWriter:
    def __init__(self):
        self.sections = []  # (name, code, bytes, count)

    def add(self, name, array):
        a = np.ascontiguousarray(array)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        key = (a.dtype.kind, a.dtype.itemsize)
        code = {("f", 4): 0, ("i", 1): 1, ("u", 1): 2, ("i", 2): 3, ("i", 4): 4, ("u", 4): 5}.get(key)
        if code is None:
            raise TypeError(f"unsupported dtype {a.dtype} for section {name}")
        self._add(name, code, a.astype(DTYPES[code]).tobytes(), a.size)

    def add_bytes(self, name, data):
        self._add(name, BYTES, bytes(data), len(data))

    def add_string(self, name, text):
        self.add_bytes(name, text.encode("utf-8") + b"\0")

    def _add(self, name, code, data, count):
        if len(name.encode()) >= NAME_SIZE:
            raise ValueError(f"section name too long: {name}")
        self.sections.append((name, code, data, count))

    def build(self):
        pos = _align(HEADER_SIZE + ENTRY_SIZE * len(self.sections))
        offsets = []
        for _, _, data, _ in self.sections:
            offsets.append(pos)
            pos = _align(pos + len(data))
        out = bytearray(pos)
        for j, (name, code, data, count) in enumerate(self.sections):
            e = HEADER_SIZE + j * ENTRY_SIZE
            out[e:e + NAME_SIZE] = name.encode().ljust(NAME_SIZE, b"\0")
            out[e + NAME_SIZE] = code
            struct.pack_into("<III", out, e + NAME_SIZE + 4, offsets[j], len(data), count)
            out[offsets[j]:offsets[j] + len(data)] = data
        struct.pack_into("<4sHHI", out, 0, b"TSTT", VERSION, len(self.sections), len(out))
        struct.pack_into("<I", out, 12, zlib.crc32(bytes(out[HEADER_SIZE:])) & 0xFFFFFFFF)
        return bytes(out)

    def write(self, path):
        data = self.build()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)


def read_container(data):
    """Returns {name: numpy array (or bytes)}; verifies magic, version and CRC."""
    magic, version, count, total = struct.unpack_from("<4sHHI", data, 0)
    if magic != b"TSTT" or version != VERSION:
        raise ValueError("not a TinySTT container")
    (crc,) = struct.unpack_from("<I", data, 12)
    if zlib.crc32(data[HEADER_SIZE:total]) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")
    out = {}
    for j in range(count):
        e = HEADER_SIZE + j * ENTRY_SIZE
        name = data[e:e + NAME_SIZE].split(b"\0")[0].decode()
        code = data[e + NAME_SIZE]
        offset, size, n = struct.unpack_from("<III", data, e + NAME_SIZE + 4)
        raw = data[offset:offset + size]
        out[name] = bytes(raw) if code == BYTES else np.frombuffer(raw, dtype=DTYPES[code]).copy()
    return out
