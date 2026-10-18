#pragma once
#include <stddef.h>
#include <stdint.h>

namespace tinystt {

/// CRC-8 (poly 0x07, init 0), used per PhonemeProtocol frame
inline uint8_t crc8(const uint8_t* data, size_t len, uint8_t crc = 0) {
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

/// CRC-32 (IEEE 802.3, reflected, as zlib.crc32), used by ModelContainer.
/// Pass the previous result as `crc` to continue a running checksum.
inline uint32_t crc32(const uint8_t* data, size_t len, uint32_t crc = 0) {
  crc = ~crc;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
  }
  return ~crc;
}

}  // namespace tinystt
