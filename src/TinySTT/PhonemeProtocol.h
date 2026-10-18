#pragma once
#include "AudioTools.h"
#include "TinySTT/Crc.h"
#include "TinySTT/PhonemeFrame.h"

namespace tinystt {

/**
 * @brief Wire format used by QueueToStream / StreamToQueue to move
 * PhonemeFrames between devices or processes (Serial, UDP, ESP-NOW, pipes).
 *
 * Layout (little endian):
 *   0xA5 | seq u8 | type u8 | index u32 | run u16 | k u8 | k x (id u8, logprob i8) | crc8
 * The CRC covers everything after the sync byte. A decoder that sees a bad
 * CRC or an impossible header drops one byte and searches for the next sync,
 * so it recovers from corrupted or truncated input.
 * See docs/protocol.md.
 */
class PhonemeProtocol {
 public:
  static constexpr uint8_t kSync = 0xA5;
  static constexpr int kHeaderSize = 1 + 1 + 1 + 4 + 2 + 1;  // through k
  static constexpr int kMaxFrameSize = kHeaderSize + 2 * PhonemeFrame::kMaxTopK + 1;

  /// Serializes one frame into `out` (>= kMaxFrameSize bytes); returns size
  int encode(const PhonemeFrame& f, uint8_t* out) {
    int k = f.k > PhonemeFrame::kMaxTopK ? PhonemeFrame::kMaxTopK : f.k;
    int n = 0;
    out[n++] = kSync;
    out[n++] = tx_seq++;
    out[n++] = f.type;
    out[n++] = (uint8_t)(f.index);
    out[n++] = (uint8_t)(f.index >> 8);
    out[n++] = (uint8_t)(f.index >> 16);
    out[n++] = (uint8_t)(f.index >> 24);
    out[n++] = (uint8_t)(f.run);
    out[n++] = (uint8_t)(f.run >> 8);
    out[n++] = (uint8_t)k;
    for (int j = 0; j < k; j++) {
      out[n++] = f.id[j];
      out[n++] = (uint8_t)f.logprob[j];
    }
    out[n] = crc8(out + 1, n - 1);
    return n + 1;
  }

  /// Writes one frame to `out`; returns false on a short write
  bool write(Print& out, const PhonemeFrame& f) {
    uint8_t buf[kMaxFrameSize];
    int n = encode(f, buf);
    return out.write(buf, n) == (size_t)n;
  }

  /// Feeds one received byte; complete frames are collected and returned
  /// by next()
  void feed(uint8_t byte) {
    if (rx_len == 0 && byte != kSync) {
      skipped_bytes++;
      return;
    }
    rx_buf[rx_len++] = byte;
    if (rx_len == kHeaderSize) {
      if (rx_buf[2] > PhonemeFrame::Reset || rx_buf[9] > PhonemeFrame::kMaxTopK) {
        resync();
        return;
      }
    }
    if (rx_len < kHeaderSize) return;
    int expected = kHeaderSize + 2 * rx_buf[9] + 1;
    if (rx_len < expected) return;

    if (crc8(rx_buf + 1, expected - 2) != rx_buf[expected - 1]) {
      crc_errors++;
      resync();
      return;
    }
    uint8_t seq = rx_buf[1];
    if (has_rx_seq && seq != (uint8_t)(last_rx_seq + 1))
      lost_frames += (uint8_t)(seq - last_rx_seq - 1);
    last_rx_seq = seq;
    has_rx_seq = true;

    PhonemeFrame frame;
    frame.type = rx_buf[2];
    frame.index = (uint32_t)rx_buf[3] | ((uint32_t)rx_buf[4] << 8) |
                  ((uint32_t)rx_buf[5] << 16) | ((uint32_t)rx_buf[6] << 24);
    frame.run = (uint16_t)(rx_buf[7] | (rx_buf[8] << 8));
    frame.k = rx_buf[9];
    for (int j = 0; j < frame.k; j++) {
      frame.id[j] = rx_buf[kHeaderSize + 2 * j];
      frame.logprob[j] = (int8_t)rx_buf[kHeaderSize + 2 * j + 1];
    }
    rx_len = 0;
    pushPending(frame);
  }

  /// Returns the next decoded frame, if any
  bool next(PhonemeFrame& frame) {
    if (pending_count == 0) return false;
    frame = pending[0];
    for (int j = 1; j < pending_count; j++) pending[j - 1] = pending[j];
    pending_count--;
    return true;
  }

  /// Convenience: feed() + next() for one byte
  bool decode(uint8_t byte, PhonemeFrame& frame) {
    feed(byte);
    return next(frame);
  }

  /// Human readable form for debugging, e.g. "#12 HH:-0.25 AH:-1.75"
  static void printText(Print& out, const PhonemeFrame& f) {
    char buf[32];
    switch (f.type) {
      case PhonemeFrame::BlankRun:
        snprintf(buf, sizeof(buf), "#%u <blank x%u>\n", (unsigned)f.index, (unsigned)f.run);
        out.print(buf);
        return;
      case PhonemeFrame::EndOfUtterance:
        snprintf(buf, sizeof(buf), "#%u <eou>\n", (unsigned)f.index);
        out.print(buf);
        return;
      case PhonemeFrame::Reset:
        out.print("<reset>\n");
        return;
      default:
        break;
    }
    snprintf(buf, sizeof(buf), "#%u", (unsigned)f.index);
    out.print(buf);
    for (int j = 0; j < f.k; j++) {
      snprintf(buf, sizeof(buf), " %s:%.2f", Phonemes::name(f.id[j]), f.logProb(j));
      out.print(buf);
    }
    out.print("\n");
  }

  void reset() {
    rx_len = 0;
    pending_count = 0;
    has_rx_seq = false;
    tx_seq = 0;
    crc_errors = lost_frames = skipped_bytes = 0;
  }

  uint32_t crcErrors() const { return crc_errors; }
  /// Frames missing according to the sequence counter
  uint32_t lostFrames() const { return lost_frames; }
  uint32_t skippedBytes() const { return skipped_bytes; }

 protected:
  uint8_t rx_buf[kMaxFrameSize];
  int rx_len = 0;
  uint8_t tx_seq = 0;
  uint8_t last_rx_seq = 0;
  bool has_rx_seq = false;
  uint32_t crc_errors = 0;
  uint32_t lost_frames = 0;
  uint32_t skipped_bytes = 0;
  // A resync can complete a frame that started inside a corrupted one, so a
  // single byte may yield more than one frame.
  static constexpr int kMaxPending = 4;
  PhonemeFrame pending[kMaxPending];
  int pending_count = 0;

  void pushPending(const PhonemeFrame& f) {
    if (pending_count == kMaxPending) {
      lost_frames++;
      return;
    }
    pending[pending_count++] = f;
  }

  /// Drop the sync byte and rescan the rest of the buffer for a new one
  void resync() {
    int old_len = rx_len;
    uint8_t tmp[kMaxFrameSize];
    memcpy(tmp, rx_buf + 1, old_len - 1);
    rx_len = 0;
    skipped_bytes++;
    for (int j = 0; j < old_len - 1; j++) feed(tmp[j]);
  }
};

}  // namespace tinystt
