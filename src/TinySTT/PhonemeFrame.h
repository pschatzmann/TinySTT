#pragma once
#include <math.h>
#include <stdint.h>

#include "TinySTT/Phonemes.h"

namespace tinystt {

/**
 * @brief The only message exchanged between SpeechToPhonemes and
 * PhonemesToText. Fixed-size POD (16 bytes), so it can be copied with memcpy
 * through the lock-free RingBufferSPSC and serialized 1:1 by PhonemeProtocol.
 *
 * A Frame carries the k best phoneme classes of one acoustic output frame
 * (30 ms) with quantized natural-log probabilities. A BlankRun stands for
 * `run` consecutive frames that were (almost) certainly CTC blank.
 */
struct PhonemeFrame {
  static constexpr int kMaxTopK = 4;
  /// logprob = q / kLogProbScale; q in [-128, 0] covers ln p in [-8, 0]
  static constexpr float kLogProbScale = 16.0f;

  enum Type : uint8_t {
    Frame = 0,
    BlankRun = 1,
    EndOfUtterance = 2,
    Reset = 3,
  };

  uint32_t index = 0;  ///< index of the (first) acoustic frame
  uint8_t type = Frame;
  uint8_t k = 0;       ///< valid entries in id[] / logprob[]
  uint16_t run = 1;    ///< number of acoustic frames represented
  uint8_t id[kMaxTopK] = {0};
  int8_t logprob[kMaxTopK] = {0};

  static int8_t quantize(float log_prob) {
    float q = roundf(log_prob * kLogProbScale);
    if (q < -128.0f) q = -128.0f;
    if (q > 0.0f) q = 0.0f;
    return (int8_t)q;
  }
  static float dequantize(int8_t q) { return q / kLogProbScale; }

  float logProb(int j) const { return dequantize(logprob[j]); }

  /// Log probability of `phoneme_id` in this frame; ids not in the top-k get
  /// the floor value (one quantization step below the k-th entry).
  float logProbOf(uint8_t phoneme_id) const {
    if (type == BlankRun) return phoneme_id == Phonemes::kBlank ? 0.0f : -8.0f;
    for (int j = 0; j < k; j++)
      if (id[j] == phoneme_id) return logProb(j);
    return k > 0 ? logProb(k - 1) - 1.0f : -8.0f;
  }

  static PhonemeFrame endOfUtterance(uint32_t idx) {
    PhonemeFrame f;
    f.type = EndOfUtterance;
    f.index = idx;
    f.run = 0;
    return f;
  }
  static PhonemeFrame reset() {
    PhonemeFrame f;
    f.type = Reset;
    f.run = 0;
    return f;
  }
  static PhonemeFrame blankRun(uint32_t idx, uint16_t frames) {
    PhonemeFrame f;
    f.type = BlankRun;
    f.index = idx;
    f.run = frames;
    return f;
  }

  bool operator==(const PhonemeFrame& o) const {
    if (index != o.index || type != o.type || k != o.k || run != o.run) return false;
    for (int j = 0; j < k; j++)
      if (id[j] != o.id[j] || logprob[j] != o.logprob[j]) return false;
    return true;
  }
  bool operator!=(const PhonemeFrame& o) const { return !(*this == o); }
};

static_assert(sizeof(PhonemeFrame) == 16, "PhonemeFrame must stay a 16 byte POD");

}  // namespace tinystt
