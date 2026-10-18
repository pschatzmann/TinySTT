#pragma once
#include <stdint.h>

#include <string>
#include <vector>

namespace tinystt {

/// One recognition hypothesis
struct SttHypothesis {
  std::string text;                 ///< words separated by single spaces
  std::vector<int> words;           ///< lexicon word ids
  std::vector<float> word_confidence;  ///< per word, 0..1
  float score = 0.0f;               ///< total log score (acoustic + LM + bonuses)
  float confidence = 0.0f;          ///< posterior of this hypothesis in the final beam, 0..1
};

/// Result reported by PhonemesToText / CtcBeamDecoder
struct SttResult {
  bool is_final = false;   ///< false: partial result while speaking
  bool accepted = true;    ///< false: final result below the minimum confidence
  uint32_t start_frame = 0;  ///< first acoustic frame (30 ms units) of the utterance
  uint32_t end_frame = 0;    ///< last acoustic frame seen
  std::vector<SttHypothesis> nbest;  ///< best first; empty if nothing recognized

  const char* text() const { return nbest.empty() ? "" : nbest[0].text.c_str(); }
  float confidence() const { return nbest.empty() ? 0.0f : nbest[0].confidence; }
};

}  // namespace tinystt
