#pragma once
// Synthetic acoustic posteriors for decoder tests: a word sequence becomes a
// sequence of log-probability frames (what the acoustic model would output),
// optionally with confusion noise. Frames go through the real FrameBuilder.
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

#include "TinySTT/FrameBuilder.h"
#include "TinySTT/Lexicon.h"

struct Synth {
  const tinystt::Lexicon& lex;
  std::mt19937 rng{123};
  float confusion = 0.0f;   ///< probability that a phone frame is confused
  float peak = 0.85f;       ///< probability of the true class
  int frames_per_phone = 2;

  explicit Synth(const tinystt::Lexicon& l) : lex(l) {}

  static std::vector<float> dist(int best, float p, int second = -1, float p2 = 0.0f) {
    std::vector<float> v(tinystt::Phonemes::kCount, 0.0f);
    float rest = 1.0f - p - p2;
    for (auto& x : v) x = rest / (tinystt::Phonemes::kCount - 1 - (second >= 0));
    v[best] = p;
    if (second >= 0) v[second] = p2;
    for (auto& x : v) x = std::log(x);
    return v;
  }

  /// Frames for a sentence (words must exist in the lexicon)
  std::vector<std::vector<float>> sentence(const std::string& text) {
    std::vector<std::vector<float>> out;
    std::uniform_real_distribution<float> u(0, 1);
    std::uniform_int_distribution<int> ph(tinystt::Phonemes::kFirstPhone, tinystt::Phonemes::kCount - 1);
    auto blank = [&](int n) {
      for (int j = 0; j < n; j++) out.push_back(dist(tinystt::Phonemes::kBlank, 0.97f));
    };
    blank(3);
    std::istringstream in(text);
    std::string w;
    while (in >> w) {
      int id = lex.lookup(w.c_str());
      uint8_t p[32];
      int n = lex.pronunciation(id, p, 32);
      for (int j = 0; j < n; j++) {
        for (int f = 0; f < frames_per_phone; f++) {
          if (u(rng) < confusion) {
            int other = ph(rng);
            if (other == p[j]) other = other == tinystt::Phonemes::kCount - 1 ? 2 : other + 1;
            out.push_back(dist(other, 0.5f, p[j], 0.35f));  // true phone is second best
          } else {
            out.push_back(dist(p[j], peak, tinystt::Phonemes::kBlank, (1 - peak) * 0.6f));
          }
        }
        blank(1);
      }
      blank(2);
    }
    blank(3);
    return out;
  }
};

/// Runs frames through FrameBuilder into a queue and returns the messages
inline std::vector<tinystt::PhonemeFrame> toMessages(const std::vector<std::vector<float>>& frames,
                                                     float blank_threshold = 0.95f, int top_k = 4) {
  tinystt::PhonemeQueue q(4096);
  tinystt::PhonemeQueueWriter w(q);
  tinystt::FrameBuilder fb(w);
  fb.setBlankThreshold(blank_threshold);
  fb.setTopK(top_k);
  for (auto& f : frames) fb.write(f.data(), (int)f.size());
  fb.endOfUtterance();
  std::vector<tinystt::PhonemeFrame> out;
  tinystt::PhonemeFrame m;
  while (q.read(m)) out.push_back(m);
  return out;
}
