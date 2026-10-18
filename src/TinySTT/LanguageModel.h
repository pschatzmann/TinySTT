#pragma once
#include <math.h>

#include "TinySTT/ModelContainer.h"

namespace tinystt {

/**
 * @brief Bigram backoff language model over the lexicon's word ids, read
 * zero-copy from the ModelContainer (built by research/build_lm.py).
 *
 * Vocabulary: lexicon words 0..W-1, then <s> = W, </s> = W+1, <unk> = W+2.
 * Word ids >= W that are not specials (overlay words from Lexicon::addWord)
 * are scored as <unk>. Without LM sections the model is uniform.
 *
 *   log p(w | h) = bigram(h, w)                 if stored
 *                = backoff(h) + unigram(w)      otherwise
 * Values are natural logs stored as int16 x scale.
 */
class LanguageModel {
 public:
  /// Binds the LM sections; returns false (and stays uniform) if missing
  bool begin(const ModelContainer& mc, int lexiconWords) {
    end();
    lexicon_words = lexiconWords;
    auto cfg = mc.find("lm.config");
    auto uni = mc.find("lm.uni_logp");
    auto bo = mc.find("lm.uni_backoff");
    auto off = mc.find("lm.bi_offsets");
    auto bw = mc.find("lm.bi_words");
    auto bl = mc.find("lm.bi_logp");
    if (!cfg.u32() || cfg.count < 4 || cfg.u32()[0] != 1 || !uni.i16() || !bo.i16() || !off.u32() ||
        !bw.valid() || !bl.valid()) {
      LOGW("TinySTT: no language model - using a uniform LM");
      return false;
    }
    uint32_t v = cfg.u32()[1];
    if ((int)v != lexiconWords + 3 || uni.count != v || bo.count != v || off.count != v + 1 ||
        bw.count != bl.count || (bw.count > 0 && (!bw.u32() || !bl.i16()))) {
      LOGE("TinySTT: language model does not match the lexicon (%u vs %d words)", (unsigned)v,
           lexiconWords + 3);
      return false;
    }
    vocab = v;
    inv_scale = 1.0f / (float)cfg.u32()[3];
    p_uni = uni.i16();
    p_backoff = bo.i16();
    p_offsets = off.u32();
    p_words = bw.u32();
    p_logp = bl.i16();
    return true;
  }

  void end() {
    p_uni = nullptr;
    vocab = 0;
  }

  bool isUniform() const { return p_uni == nullptr; }

  /// Sets the lexicon size used by the uniform model and the special ids
  void setLexiconWords(int n) {
    if (isUniform()) lexicon_words = n;
  }

  int bos() const { return lexicon_words; }
  int eos() const { return lexicon_words + 1; }
  int unk() const { return lexicon_words + 2; }

  /// Natural log probability of `word` after `prev` (use bos() at the start)
  float logProb(int prev, int word) const {
    if (isUniform()) return -logf((float)(lexicon_words + 2));
    int h = mapId(prev), w = mapId(word);
    uint32_t lo = p_offsets[h], hi = p_offsets[h + 1];
    // binary search in the sorted successors of h
    while (lo < hi) {
      uint32_t mid = (lo + hi) / 2;
      if (p_words[mid] < (uint32_t)w)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo < p_offsets[h + 1] && p_words[lo] == (uint32_t)w) return p_logp[lo] * inv_scale;
    return (p_backoff[h] + p_uni[w]) * inv_scale;
  }

  /// log p(</s> | prev)
  float endLogProb(int prev) const { return logProb(prev, eos()); }

  uint32_t vocabSize() const { return isUniform() ? (uint32_t)lexicon_words + 3 : vocab; }

 protected:
  int lexicon_words = 0;
  uint32_t vocab = 0;
  float inv_scale = 0.001f;
  const int16_t* p_uni = nullptr;
  const int16_t* p_backoff = nullptr;
  const uint32_t* p_offsets = nullptr;
  const uint32_t* p_words = nullptr;
  const int16_t* p_logp = nullptr;

  int mapId(int id) const {
    if (id >= 0 && id < lexicon_words + 3 && id != unk()) return id;
    return unk();
  }
};

}  // namespace tinystt
