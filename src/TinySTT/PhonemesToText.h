#pragma once
#include <string>
#include <vector>

#include "AudioTools.h"
#include "TinySTT/CtcBeamDecoder.h"
#include "TinySTT/Grammar.h"
#include "TinySTT/LanguageModel.h"
#include "TinySTT/Lexicon.h"
#include "TinySTT/PhonemeFrame.h"

namespace tinystt {

/**
 * @brief Stage 2: phonemes -> text. The single consumer of the phoneme queue:
 * copy() drains it into the CtcBeamDecoder and reports results through a
 * callback and/or prints final texts (one line each) to a Print.
 *
 * Modes: free dictation (lexicon + LM, default), word list
 * (setWordListMode(true) + addWord()), or grammar (setGrammar()).
 */
class PhonemesToText {
 public:
  using ResultCallback = CtcBeamDecoder::ResultCallback;

  explicit PhonemesToText(audio_tools::BaseBuffer<PhonemeFrame>& queue) : p_queue(&queue) {}

  /// Lexicon and LM from the container (both optional: without a lexicon
  /// only words added with addWord()/setGrammar() are recognized)
  bool begin(const ModelContainer& mc) {
    lex.begin(mc);  // a container without lexicon is fine (word list / grammar)
    lm.begin(mc, lex.staticWordCount());
    lm.setLexiconWords(lex.staticWordCount());
    return beginDecoder();
  }

  /// Without any container: word-list / grammar use only
  bool begin() {
    lm.setLexiconWords(0);
    return beginDecoder();
  }

  void setResultCallback(ResultCallback cb, void* ref = nullptr) {
    user_callback = cb;
    user_ref = ref;
  }
  /// Prints final, accepted results as text lines
  void setOutput(Print& out) { p_out = &out; }

  /// Drains the queue; returns the number of messages processed
  size_t copy() {
    size_t n = 0;
    PhonemeFrame f;
    while (p_queue->read(f)) {
      dec.write(f);
      n++;
    }
    return n;
  }

  // ---- vocabulary ----
  // Words added before begin() are remembered and added in begin(), when the
  // dictionary is loaded (so they get its pronunciation); then -1 is returned.

  /// Adds a word (dictionary pronunciation, else letter-to-sound rules)
  int addWord(const char* word) { return addHotword(word, 0.0f); }
  /// Adds a word with an explicit pronunciation (phoneme ids)
  int addWord(const char* word, const uint8_t* phones, int n) { return lex.addWord(word, phones, n); }
  /// Adds (or boosts) a word; boost is a log-domain bonus (e.g. 2..5)
  int addHotword(const char* word, float boost = 3.0f) {
    if (!started) {
      pending_words.push_back({word, boost});
      return -1;
    }
    return lex.addWord(word, boost);
  }
  /// true: only words added with addWord()/addHotword() are recognized.
  /// Can be called before or after begin().
  void setWordListMode(bool flag) {
    word_list_mode = flag;
    applyMode();
  }

  /**
   * Grammar (command) mode; the grammar words are added to the lexicon
   * overlay. Can be called before begin() (applied there) or after it; with
   * running tasks, change it only while no audio is being decoded.
   * Returns false on a syntax error (see grammar().error()).
   */
  bool setGrammar(const char* grammar, bool wordsOnly = true) {
    if (!started) {
      // check the syntax now; compile with the dictionary in begin()
      Lexicon scratch;
      if (!gram.compile(grammar, scratch, false)) return false;
      pending_grammar = grammar;
    } else if (!gram.compile(grammar, lex)) {
      return false;
    }
    grammar_active = true;
    grammar_words_only = wordsOnly;
    applyMode();
    return true;
  }
  /// Back to the language model (free dictation)
  void clearGrammar() {
    grammar_active = false;
    applyMode();
  }

  CtcBeamDecoder& decoder() { return dec; }
  CtcBeamDecoder::Config& decoderConfig() { return dec.config(); }
  Lexicon& lexicon() { return lex; }
  LanguageModel& languageModel() { return lm; }
  const Grammar& grammar() const { return gram; }
  const SttResult& lastResult() const { return last_final; }

 protected:
  audio_tools::BaseBuffer<PhonemeFrame>* p_queue;
  Lexicon lex;
  LanguageModel lm;
  LanguageModelScorer lm_scorer;
  Grammar gram;
  CtcBeamDecoder dec;
  ResultCallback user_callback = nullptr;
  void* user_ref = nullptr;
  Print* p_out = nullptr;
  SttResult last_final;

  bool word_list_mode = false;
  bool grammar_active = false;
  bool grammar_words_only = true;
  bool started = false;

  std::vector<std::pair<std::string, float>> pending_words;
  std::string pending_grammar;

  bool beginDecoder() {
    lm_scorer.setLanguageModel(lm);
    dec.begin(lex, lm_scorer);
    dec.setResultCallback(onResult, this);
    started = true;
    // settings made before begin(), now with the dictionary loaded
    for (auto& w : pending_words) lex.addWord(w.first.c_str(), w.second);
    pending_words.clear();
    if (!pending_grammar.empty()) {
      if (!gram.compile(pending_grammar.c_str(), lex)) grammar_active = false;
      pending_grammar.clear();
    }
    applyMode();
    return true;
  }

  /// Applies word-list / grammar settings (also those made before begin())
  void applyMode() {
    bool only_overlay = word_list_mode || (grammar_active && grammar_words_only);
    lex.setUseStatic(!only_overlay);
    if (!started) return;
    dec.setWordScorer(grammar_active ? (const WordScorer&)gram : (const WordScorer&)lm_scorer);
    dec.reset();
  }

  static void onResult(const SttResult& r, void* ref) {
    PhonemesToText* self = (PhonemesToText*)ref;
    if (r.is_final) {
      self->last_final = r;
      if (self->p_out != nullptr && r.accepted && !r.nbest.empty()) self->p_out->println(r.text());
    }
    if (self->user_callback) self->user_callback(r, self->user_ref);
  }
};

}  // namespace tinystt
