#pragma once
#include "TinySTT/LanguageModel.h"

namespace tinystt {

/**
 * @brief Word-level constraint/score used by the decoder: a language model
 * (free dictation) or a grammar (command mode). The decoder keeps one integer
 * state per hypothesis.
 */
class WordScorer {
 public:
  virtual ~WordScorer() = default;
  /// State at the start of an utterance
  virtual int startState() const = 0;
  /// Score (natural log) of `word` in `state`; false if the word is not
  /// allowed. On success `next` receives the new state.
  virtual bool next(int state, int word, int& next, float& score) const = 0;
  /// Score for ending the utterance in `state`; false if not a final state
  virtual bool end(int state, float& score) const = 0;
};

/// WordScorer backed by the bigram LanguageModel (state = previous word id)
class LanguageModelScorer : public WordScorer {
 public:
  LanguageModelScorer() = default;
  explicit LanguageModelScorer(const LanguageModel& lm) : p_lm(&lm) {}
  void setLanguageModel(const LanguageModel& lm) { p_lm = &lm; }

  int startState() const override { return p_lm->bos(); }
  bool next(int state, int word, int& next, float& score) const override {
    // lexicon overlay words (ids >= static word count) share their id range
    // with the LM specials <s> </s> <unk>: score them as <unk>
    int id = word < p_lm->bos() ? word : p_lm->unk();
    score = p_lm->logProb(state, id);
    next = id;
    return true;
  }
  bool end(int state, float& score) const override {
    score = p_lm->endLogProb(state);
    return true;
  }

 protected:
  const LanguageModel* p_lm = nullptr;
};

}  // namespace tinystt
