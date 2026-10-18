#pragma once
#include <stdint.h>
#include <string.h>

#include <vector>

#include "AudioTools.h"
#include "TinySTT/Crc.h"

namespace tinystt {

/**
 * @brief One versioned binary file for all model data: acoustic weights,
 * lexicon, language model and grammars. Written by research/model_container.py
 * (and ModelContainerWriter for tests), read zero-copy from PROGMEM, a
 * memory-mapped flash partition, or loaded from a Stream/File into RAM/PSRAM.
 *
 * Layout (little endian, see docs/model-data.md):
 *   Header  16 B: magic "TSTT" | version u16 | section_count u16 |
 *                 total_size u32 | crc32 u32 (of bytes [16, total_size))
 *   Section table: section_count x 48 B:
 *                 name char[32] (NUL padded) | dtype u8 | pad[3] |
 *                 offset u32 (from file start) | size u32 (bytes) | count u32
 *   Section data, each 16 byte aligned.
 */
class ModelContainer {
 public:
  static constexpr uint32_t kMagic = 0x54545354;  // "TSTT" read as LE u32
  static constexpr uint16_t kVersion = 1;
  static constexpr int kHeaderSize = 16;
  static constexpr int kEntrySize = 48;
  static constexpr int kNameSize = 32;
  static constexpr int kAlign = 16;

  enum DType : uint8_t { F32 = 0, I8 = 1, U8 = 2, I16 = 3, I32 = 4, U32 = 5, Bytes = 6 };

  enum Error {
    OK = 0,
    TooSmall,
    BadMagic,
    BadVersion,
    BadSize,
    BadCrc,
    BadSection,
    ReadError,
    NoMemory,
  };

  struct Section {
    const char* name = nullptr;  ///< points into the container
    DType dtype = Bytes;
    const uint8_t* data = nullptr;
    uint32_t size = 0;   ///< bytes
    uint32_t count = 0;  ///< elements

    bool valid() const { return data != nullptr; }
    const float* f32() const { return dtype == F32 ? (const float*)data : nullptr; }
    const int8_t* i8() const { return dtype == I8 ? (const int8_t*)data : nullptr; }
    const uint8_t* u8() const { return data; }
    const int16_t* i16() const { return dtype == I16 ? (const int16_t*)data : nullptr; }
    const int32_t* i32() const { return dtype == I32 ? (const int32_t*)data : nullptr; }
    const uint32_t* u32() const { return dtype == U32 ? (const uint32_t*)data : nullptr; }
  };

  static size_t dtypeSize(DType t) {
    switch (t) {
      case F32: case I32: case U32: return 4;
      case I16: return 2;
      default: return 1;
    }
  }

  ModelContainer() = default;
  ModelContainer(const ModelContainer&) = delete;
  ModelContainer& operator=(const ModelContainer&) = delete;

  /// Uses the data in place (PROGMEM array, mmapped partition, buffer that
  /// outlives this object). The data must be 4 byte aligned.
  Error begin(const uint8_t* data, size_t len, bool verifyCrc = true) {
    owned.clear();
    return parse(data, len, verifyCrc);
  }

  /// Loads `len` bytes from a Stream/File into memory owned by this object.
  /// On ESP32 with PSRAM, pass AllocatorPSRAM to keep it out of internal RAM.
  Error begin(Stream& in, size_t len, bool verifyCrc = true) {
    owned.resize(len);
    if (owned.size() != len) return setError(NoMemory);
    size_t total = 0;
    while (total < len) {
      size_t n = in.readBytes(owned.data() + total, len - total);
      if (n == 0) return setError(ReadError);
      total += n;
    }
    return parse(owned.data(), len, verifyCrc);
  }

  void end() {
    owned.clear();
    p_data = nullptr;
    data_len = 0;
    section_count = 0;
  }

  operator bool() const { return p_data != nullptr && last_error == OK; }
  Error error() const { return last_error; }
  uint16_t version() const { return file_version; }
  int sectionCount() const { return section_count; }
  size_t size() const { return data_len; }

  Section section(int idx) const {
    Section s;
    if (p_data == nullptr || idx < 0 || idx >= section_count) return s;
    const uint8_t* e = p_data + kHeaderSize + idx * kEntrySize;
    s.name = (const char*)e;
    s.dtype = (DType)e[kNameSize];
    uint32_t offset = rd32(e + kNameSize + 4);
    s.size = rd32(e + kNameSize + 8);
    s.count = rd32(e + kNameSize + 12);
    s.data = p_data + offset;
    return s;
  }

  /// Section by name; Section::valid() is false if missing
  Section find(const char* name) const {
    for (int j = 0; j < section_count; j++) {
      Section s = section(j);
      if (strncmp(s.name, name, kNameSize) == 0) return s;
    }
    return Section();
  }

  bool has(const char* name) const { return find(name).valid(); }

  static uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  }
  static uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

 protected:
  std::vector<uint8_t> owned;
  const uint8_t* p_data = nullptr;
  size_t data_len = 0;
  int section_count = 0;
  uint16_t file_version = 0;
  Error last_error = OK;

  Error setError(Error e) {
    last_error = e;
    if (e != OK) {
      p_data = nullptr;
      section_count = 0;
      LOGE("TinySTT: invalid model container (error %d)", (int)e);
    }
    return e;
  }

  Error parse(const uint8_t* data, size_t len, bool verifyCrc) {
    p_data = nullptr;
    if (data == nullptr || len < (size_t)kHeaderSize) return setError(TooSmall);
    if (rd32(data) != kMagic) return setError(BadMagic);
    file_version = rd16(data + 4);
    if (file_version != kVersion) return setError(BadVersion);
    int count = rd16(data + 6);
    uint32_t total = rd32(data + 8);
    if (total > len || total < (uint32_t)(kHeaderSize + count * kEntrySize)) return setError(BadSize);
    if (verifyCrc && crc32(data + kHeaderSize, total - kHeaderSize) != rd32(data + 12))
      return setError(BadCrc);
    for (int j = 0; j < count; j++) {
      const uint8_t* e = data + kHeaderSize + j * kEntrySize;
      uint32_t offset = rd32(e + kNameSize + 4);
      uint32_t size = rd32(e + kNameSize + 8);
      uint8_t dtype = e[kNameSize];
      if (dtype > Bytes || (uint64_t)offset + size > total) return setError(BadSection);
      if (e[kNameSize - 1] != 0) return setError(BadSection);  // name must be NUL terminated
    }
    p_data = data;
    data_len = total;
    section_count = count;
    return setError(OK);
  }
};

/**
 * @brief Builds a ModelContainer image in memory (host tools and tests; the
 * Python exporter writes the identical format).
 */
class ModelContainerWriter {
 public:
  void add(const char* name, ModelContainer::DType dtype, const void* data, size_t bytes) {
    Entry e;
    strncpy(e.name, name, ModelContainer::kNameSize - 1);
    e.name[ModelContainer::kNameSize - 1] = 0;
    e.dtype = dtype;
    e.data.assign((const uint8_t*)data, (const uint8_t*)data + bytes);
    e.count = (uint32_t)(bytes / ModelContainer::dtypeSize(dtype));
    entries.push_back(e);
  }
  void addF32(const char* name, const float* v, size_t n) { add(name, ModelContainer::F32, v, n * 4); }
  void addI8(const char* name, const int8_t* v, size_t n) { add(name, ModelContainer::I8, v, n); }
  void addU32(const char* name, const uint32_t* v, size_t n) { add(name, ModelContainer::U32, v, n * 4); }
  void addBytes(const char* name, const void* v, size_t n) { add(name, ModelContainer::Bytes, v, n); }
  void addString(const char* name, const char* str) { addBytes(name, str, strlen(str) + 1); }

  std::vector<uint8_t> build() const {
    using MC = ModelContainer;
    size_t pos = align(MC::kHeaderSize + entries.size() * MC::kEntrySize);
    std::vector<uint32_t> offsets;
    for (auto& e : entries) {
      offsets.push_back((uint32_t)pos);
      pos = align(pos + e.data.size());
    }
    std::vector<uint8_t> out(pos, 0);
    wr32(&out[0], MC::kMagic);
    wr16(&out[4], MC::kVersion);
    wr16(&out[6], (uint16_t)entries.size());
    wr32(&out[8], (uint32_t)out.size());
    for (size_t j = 0; j < entries.size(); j++) {
      uint8_t* e = &out[MC::kHeaderSize + j * MC::kEntrySize];
      memcpy(e, entries[j].name, MC::kNameSize);
      e[MC::kNameSize] = entries[j].dtype;
      wr32(e + MC::kNameSize + 4, offsets[j]);
      wr32(e + MC::kNameSize + 8, (uint32_t)entries[j].data.size());
      wr32(e + MC::kNameSize + 12, entries[j].count);
      if (!entries[j].data.empty())
        memcpy(&out[offsets[j]], entries[j].data.data(), entries[j].data.size());
    }
    wr32(&out[12], crc32(&out[MC::kHeaderSize], out.size() - MC::kHeaderSize));
    return out;
  }

  void clear() { entries.clear(); }

 protected:
  struct Entry {
    char name[ModelContainer::kNameSize] = {0};
    ModelContainer::DType dtype = ModelContainer::Bytes;
    std::vector<uint8_t> data;
    uint32_t count = 0;
  };
  std::vector<Entry> entries;

  static size_t align(size_t v) { return (v + ModelContainer::kAlign - 1) & ~(size_t)(ModelContainer::kAlign - 1); }
  static void wr32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
  }
  static void wr16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
};

}  // namespace tinystt
