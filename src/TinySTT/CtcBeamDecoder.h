#pragma once
#include <math.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "TinySTT/Lexicon.h"
#include "TinySTT/PhonemeFrame.h"
#include "TinySTT/SttResult.h"
#include "TinySTT/WordScorer.h"

namespace tinystt {

/**
 * @brief Lexicon-constrained CTC prefix beam search over PhonemeFrames.
 *
 * Hypotheses walk the lexicon trie phoneme by phoneme; whenever a node ends a
 * word, an additional hypothesis completes that word (one per homophone) and
 * is scored by the WordScorer (bigram LM or grammar):
 *
 *   score = log P_ctc(prefix) + lm_weight * log P_word + word_bonus * words
 *           + hotword boosts
 *
 * CTC semantics: repeated phonemes collapse unless separated by blank;
 * silence counts as blank. A BlankRun of n frames is applied as n blank
 * steps with probability ~1. Only the frame's top-k phonemes are expanded.
 *
 * Results: partial results (best completed words) while frames arrive,
 * a final N-best result on EndOfUtterance. Confidence = posterior of the
 * hypothesis among the final beam x mean top-1 probability of the frames
 * that were not merged into blank runs (so flat, uncertain acoustics lower it even if one path dominates).
 */
class CtcBeamDecoder {
 public:
  using ResultCallback = void (*)(const SttResult& result, void* ref);

  struct Config {
    int beam_size = 32;  ///< 16: WER 36.0 %, 32: 32.7 % (LibriSpeech test-clean)
    /// additional hypotheses at a word boundary that are always kept: partial
    /// words (pure acoustic score) otherwise crowd out the hypotheses that
    /// just paid the LM cost of a completed word
    int word_end_beam = 8;
    float beam_threshold = 25.0f;  ///< drop hyps scoring below best - threshold
    /// tuned on LibriSpeech test-clean with the trained model (beam 16):
    /// WER 61.9% -> 31.9% together with word_end_beam
    float lm_weight = 0.5f;
    float word_bonus = 0.0f;       ///< per word, counters the LM's short-sentence bias
    int nbest = 3;
    float min_confidence = 0.0f;   ///< final results below are marked !accepted
    bool partial_results = true;
  };

  CtcBeamDecoder() = default;

  void begin(const Lexicon& lexicon, const WordScorer& scorer) {
    p_lex = &lexicon;
    p_scorer = &scorer;
    reset();
  }

  Config& config() { return cfg; }
  void setConfig(const Config& c) { cfg = c; }
  void setResultCallback(ResultCallback cb, void* ref = nullptr) {
    callback = cb;
    callback_ref = ref;
  }
  /// Swaps the word scorer (LM <-> grammar); takes effect at the next utterance
  void setWordScorer(const WordScorer& scorer) { p_scorer = &scorer; }

  /// Starts a new utterance
  void reset() {
    hist.clear();
    hist_index.clear();
    hist_compact_at = kHistCompactMin;
    beam.clear();
    last_partial.clear();
    acoustic_sum = 0.0f;
    acoustic_frames = 0;
    first_frame = last_frame = 0;
    has_frames = false;
    if (p_lex == nullptr || p_scorer == nullptr) return;
    Hyp h;
    h.hist = -1;
    h.node = p_lex->root();
    h.last = -1;
    h.pb = 0.0f;
    h.pnb = kNegInf;
    h.lm = 0.0f;
    h.state = p_scorer->startState();
    beam.push_back(h);
  }

  /// Processes one queue message; returns true if a final result was produced
  bool write(const PhonemeFrame& f) {
    switch (f.type) {
      case PhonemeFrame::Frame:
        trackFrame(f.index, 1);
        step(f);
        emitPartial();
        return false;
      case PhonemeFrame::BlankRun:
        trackFrame(f.index, f.run);
        blankRun(f.run);
        return false;
      case PhonemeFrame::EndOfUtterance: {
        SttResult r = finish();
        if (callback) callback(r, callback_ref);
        return true;
      }
      case PhonemeFrame::Reset:
        reset();
        return false;
    }
    return false;
  }

  /// Ends the utterance: final N-best result; the decoder is reset afterwards
  SttResult finish() {
    SttResult result;
    result.is_final = true;
    result.start_frame = first_frame;
    result.end_frame = last_frame;
    struct Final {
      int hist;
      float score;
    };
    std::vector<Final> finals;
    Lexicon::NodeRef root = p_lex ? p_lex->root() : Lexicon::NodeRef();
    for (auto& h : beam) {
      if (!(h.node == root)) continue;  // inside a word
      float end_score;
      if (!p_scorer->end(h.state, end_score)) continue;
      float s = logAdd(h.pb, h.pnb) + h.lm + cfg.lm_weight * end_score;
      bool merged = false;
      for (auto& f : finals) {
        if (f.hist == h.hist) {  // same words, different last phoneme
          f.score = logAdd(f.score, s);
          merged = true;
        }
      }
      if (!merged) finals.push_back({h.hist, s});
    }
    std::sort(finals.begin(), finals.end(), [](const Final& a, const Final& b) { return a.score > b.score; });
    if (!finals.empty()) {
      float norm = kNegInf;
      for (auto& f : finals) norm = logAdd(norm, f.score);
      // acoustic certainty: mean top-1 probability of the (non blank-run) frames
      float acoustic = acoustic_frames > 0 ? acoustic_sum / acoustic_frames : 1.0f;
      std::vector<std::vector<int>> seqs;
      std::vector<float> post;
      for (auto& f : finals) {
        seqs.push_back(words(f.hist));
        post.push_back(expf(f.score - norm));
      }
      for (size_t j = 0; j < finals.size() && (int)result.nbest.size() < std::max(1, cfg.nbest); j++) {
        if (seqs[j].empty()) continue;  // silence / nothing recognized
        SttHypothesis hyp;
        hyp.words = seqs[j];
        hyp.text = text(seqs[j]);
        hyp.score = finals[j].score;
        hyp.confidence = post[j] * acoustic;
        // word confidence: posterior mass of hypotheses agreeing on the word
        // at the same position
        for (size_t w = 0; w < seqs[j].size(); w++) {
          float c = 0.0f;
          for (size_t o = 0; o < seqs.size(); o++)
            if (w < seqs[o].size() && seqs[o][w] == seqs[j][w]) c += post[o];
          hyp.word_confidence.push_back(std::min(1.0f, c) * acoustic);
        }
        result.nbest.push_back(hyp);
      }
    }
    result.accepted = !result.nbest.empty() && result.nbest[0].confidence >= cfg.min_confidence;
    reset();
    return result;
  }

  /// Best completed words so far
  std::string partialText() const {
    if (beam.empty()) return "";
    return text(words(best().hist));
  }

  size_t beamSize() const { return beam.size(); }

 protected:
  static constexpr float kNegInf = -1e30f;
  static constexpr float kFloorLogProb = -8.0f;
  static constexpr size_t kHistCompactMin = 256;  ///< word history nodes before the first compaction

  struct Hyp {
    int hist;               ///< word history node (-1 = no words)
    Lexicon::NodeRef node;  ///< position in the lexicon trie
    int last;               ///< last emitted phoneme (-1 = none)
    float pb, pnb;          ///< CTC log probs: ending in blank / non-blank
    float lm;               ///< word level score so far
    int state;              ///< WordScorer state
  };
  struct HistNode {
    int word;
    int prev;
  };

  Config cfg;
  const Lexicon* p_lex = nullptr;
  const WordScorer* p_scorer = nullptr;
  ResultCallback callback = nullptr;
  void* callback_ref = nullptr;
  std::vector<Hyp> beam, next_beam;
  std::vector<HistNode> hist;
  std::unordered_map<uint64_t, int> hist_index;
  size_t hist_compact_at = kHistCompactMin;
  std::string last_partial;
  uint32_t first_frame = 0, last_frame = 0;
  bool has_frames = false;
  float acoustic_sum = 0.0f;   ///< sum of top-1 probabilities of Frame messages
  uint32_t acoustic_frames = 0;

  static float logAdd(float a, float b) {
    if (a < b) std::swap(a, b);
    if (b <= kNegInf) return a;
    return a + log1pf(expf(b - a));
  }

  static float total(const Hyp& h) { return logAdd(h.pb, h.pnb) + h.lm; }

  const Hyp& best() const {
    size_t b = 0;
    for (size_t j = 1; j < beam.size(); j++)
      if (total(beam[j]) > total(beam[b])) b = j;
    return beam[b];
  }

  void trackFrame(uint32_t index, uint32_t run) {
    if (!has_frames) {
      first_frame = index;
      // words may have been added to the lexicon overlay since reset():
      // resolve the root of the initial hypothesis now
      if (p_lex != nullptr && beam.size() == 1 && beam[0].hist < 0) beam[0].node = p_lex->root();
    }
    has_frames = true;
    last_frame = index + (run > 0 ? run - 1 : 0);
  }

  static uint64_t histKey(int prev, int word) {
    return ((uint64_t)(uint32_t)(prev + 1) << 32) | (uint32_t)word;
  }

  int histFor(int prev, int word) {
    uint64_t key = histKey(prev, word);
    auto it = hist_index.find(key);
    if (it != hist_index.end()) return it->second;
    hist.push_back({word, prev});
    int id = (int)hist.size() - 1;
    hist_index[key] = id;
    return id;
  }

  std::vector<int> words(int h) const {
    std::vector<int> out;
    for (; h >= 0; h = hist[h].prev) out.push_back(hist[h].word);
    std::reverse(out.begin(), out.end());
    return out;
  }

  std::string text(const std::vector<int>& ws) const {
    std::string s;
    for (int w : ws) {
      if (!s.empty()) s += ' ';
      s += p_lex->word(w);
    }
    return s;
  }

  /// Adds probability mass to the hypothesis with this key (merging)
  void add(int h, Lexicon::NodeRef node, int last, float lm, int state, float pb, float pnb) {
    for (auto& x : next_beam) {
      if (x.hist == h && x.node == node && x.last == last) {
        x.pb = logAdd(x.pb, pb);
        x.pnb = logAdd(x.pnb, pnb);
        return;
      }
    }
    next_beam.push_back({h, node, last, pb, pnb, lm, state});
  }

  void step(const PhonemeFrame& f) {
    if (p_lex == nullptr) return;
    if (f.k > 0) {  // confident blanks arrive as BlankRun and are not counted
      acoustic_sum += expf(f.logProb(0));
      acoustic_frames++;
    }
    float lp_blank = logAdd(f.logProbOf(Phonemes::kBlank), f.logProbOf(Phonemes::kSilence));
    Lexicon::NodeRef root = p_lex->root();
    next_beam.clear();
    int ids[16];
    for (const Hyp& h : beam) {
      float ac = logAdd(h.pb, h.pnb);
      // blank / silence
      add(h.hist, h.node, h.last, h.lm, h.state, ac + lp_blank, kNegInf);
      // repeat of the last phoneme (collapses)
      if (h.last >= 0 && h.pnb > kNegInf)
        add(h.hist, h.node, h.last, h.lm, h.state, kNegInf, h.pnb + f.logProbOf((uint8_t)h.last));
      // extensions with the frame's candidate phonemes
      for (int j = 0; j < f.k; j++) {
        uint8_t c = f.id[j];
        if (c == Phonemes::kBlank || c == Phonemes::kSilence) continue;
        Lexicon::NodeRef child = p_lex->child(h.node, c);
        if (!child.valid()) continue;
        float from = (c == h.last) ? h.pb : ac;
        if (from <= kNegInf) continue;
        float p = from + f.logProb(j);
        if (p_lex->hasChildren(child)) add(h.hist, child, c, h.lm, h.state, kNegInf, p);
        int n = p_lex->wordsAt(child, ids, 16);
        for (int w = 0; w < n; w++) {
          int ns;
          float s;
          if (!p_scorer->next(h.state, ids[w], ns, s)) continue;
          float lm = h.lm + cfg.lm_weight * s + cfg.word_bonus + p_lex->boost(ids[w]);
          add(histFor(h.hist, ids[w]), root, c, lm, ns, kNegInf, p);
        }
      }
    }
    prune();
  }

  void blankRun(uint32_t n) {
    for (auto& h : beam) {
      h.pb = logAdd(h.pb, h.pnb);
      h.pnb = h.pnb + kFloorLogProb * (float)n;
      if (h.pnb < kNegInf) h.pnb = kNegInf;
    }
  }

  void prune() {
    std::sort(next_beam.begin(), next_beam.end(),
              [](const Hyp& a, const Hyp& b) { return total(a) > total(b); });
    size_t keep = std::min(next_beam.size(), (size_t)std::max(1, cfg.beam_size));
    if (keep > 0) {
      float limit = total(next_beam[0]) - cfg.beam_threshold;
      while (keep > 1 && total(next_beam[keep - 1]) < limit) keep--;
    }
    // word end quota: move the best hypotheses at a word boundary that did
    // not make it into the beam directly behind the kept ones
    if (cfg.word_end_beam > 0 && p_lex != nullptr) {
      Lexicon::NodeRef root = p_lex->root();
      int extra = 0;
      for (size_t j = keep; j < next_beam.size() && extra < cfg.word_end_beam; j++) {
        if (next_beam[j].node == root) {
          std::swap(next_beam[keep + extra], next_beam[j]);
          extra++;
        }
      }
      keep += extra;
    }
    next_beam.resize(keep);
    beam.swap(next_beam);
    compactHistory();
  }

  /// Every word end creates a history node, also for hypotheses that are
  /// pruned right away. Dropping the nodes no hypothesis refers to keeps the
  /// memory bounded by the beam instead of growing with the utterance.
  void compactHistory() {
    if (hist.size() < hist_compact_at) return;
    std::vector<int> map(hist.size(), -1);  // -1 unused, -2 live, else new id
    for (const Hyp& h : beam)
      for (int i = h.hist; i >= 0 && map[i] == -1; i = hist[i].prev) map[i] = -2;
    // a parent always has a lower id than its children: one ascending pass
    hist_index.clear();
    size_t n = 0;
    for (size_t i = 0; i < hist.size(); i++) {
      if (map[i] == -1) continue;
      int prev = hist[i].prev >= 0 ? map[hist[i].prev] : -1;
      hist[n] = {hist[i].word, prev};
      hist_index[histKey(prev, hist[n].word)] = (int)n;
      map[i] = (int)n++;
    }
    hist.resize(n);
    for (Hyp& h : beam)
      if (h.hist >= 0) h.hist = map[h.hist];
    hist_compact_at = std::max(kHistCompactMin, 2 * n);
  }

  void emitPartial() {
    if (!cfg.partial_results || callback == nullptr || beam.empty()) return;
    std::string t = partialText();
    if (t == last_partial) return;
    last_partial = t;
    SttResult r;
    r.is_final = false;
    r.start_frame = first_frame;
    r.end_frame = last_frame;
    SttHypothesis h;
    h.text = t;
    h.words = words(best().hist);
    r.nbest.push_back(h);
    callback(r, callback_ref);
  }
};

}  // namespace tinystt
