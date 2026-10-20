// Step 8 + 8b: CtcBeamDecoder (LM, word list, hotwords, N-best, confidence,
// partial results, blank skipping) and Grammar, on synthetic posteriors.
#include "AudioTools.h"
#include "Synth.h"
#include "TestData.h"
#include "TestUtil.h"
#include "TinySTT/CtcBeamDecoder.h"
#include "TinySTT/Grammar.h"

using namespace tinystt;

static std::vector<uint8_t> g_img = readTestFile("test_lexicon_lm.tstt");

struct Fixture {
  ModelContainer mc;
  Lexicon lex;
  LanguageModel lm;
  LanguageModelScorer scorer;
  CtcBeamDecoder dec;
  std::vector<SttResult> partials;
  Fixture() {
    mc.begin(g_img.data(), g_img.size());
    lex.begin(mc);
    lm.begin(mc, lex.staticWordCount());
    scorer.setLanguageModel(lm);
    dec.begin(lex, scorer);
    dec.setResultCallback(onResult, this);
  }
  static void onResult(const SttResult& r, void* ref) {
    if (!r.is_final) ((Fixture*)ref)->partials.push_back(r);
  }
  SttResult decode(const std::vector<PhonemeFrame>& msgs) {
    SttResult final_result;
    for (auto& m : msgs) {
      if (m.type == PhonemeFrame::EndOfUtterance) {
        final_result = dec.finish();
      } else {
        dec.write(m);
      }
    }
    return final_result;
  }
  SttResult decode(const std::string& text, float confusion = 0.0f, float blank_threshold = 0.95f) {
    Synth s(lex);
    s.confusion = confusion;
    return decode(toMessages(s.sentence(text), blank_threshold));
  }
};

TEST(decoder_exact_sentences) {
  Fixture fx;
  for (const char* t : {"turn on the kitchen light", "turn off the fan", "what time is it", "hello world",
                        "go to the bedroom"}) {
    auto r = fx.decode(t);
    CHECK_STR(r.text(), t);
    CHECK(r.is_final);
    CHECK(r.accepted);
  }
}

TEST(decoder_lm_resolves_homophones) {
  Fixture fx;
  CHECK_STR(fx.decode("set a timer for two minutes").text(), "set a timer for two minutes");
  CHECK_STR(fx.decode("go to the kitchen").text(), "go to the kitchen");
  CHECK_STR(fx.decode("write it right").text(), "write it right");
}

TEST(decoder_robust_to_confusions) {
  Fixture fx;
  int correct = 0;
  const char* sentences[] = {"turn on the kitchen light", "turn off the bedroom light", "please turn on the light",
                             "set a timer for four minutes"};
  for (int rep = 0; rep < 3; rep++) {
    for (const char* t : sentences) {
      Synth s(fx.lex);
      s.rng.seed(1000 + rep);
      s.confusion = 0.15f;
      auto r = fx.decode(toMessages(s.sentence(t)));
      correct += std::string(r.text()) == t;
    }
  }
  std::printf("  %d / 12 correct with 15%% confused frames\n", correct);
  CHECK(correct >= 11);
}

TEST(decoder_blank_skipping_same_result) {
  Fixture fx;
  Synth s(fx.lex);
  auto frames = s.sentence("turn on the kitchen light");
  auto with = toMessages(frames, 0.95f);
  auto without = toMessages(frames, 1.0f);
  CHECK(with.size() < without.size());
  std::printf("  %zu messages with blank skipping, %zu without\n", with.size(), without.size());
  auto a = fx.decode(with);
  auto b = fx.decode(without);
  CHECK_STR(a.text(), b.text());
  CHECK_STR(a.text(), "turn on the kitchen light");
}

/// exposes the size of the word history
struct HistDecoder : public CtcBeamDecoder {
  size_t historyNodes() const { return hist.size(); }
};

TEST(decoder_history_stays_bounded) {
  Fixture fx;
  HistDecoder dec;
  dec.begin(fx.lex, fx.scorer);
  std::string text;
  for (int j = 0; j < 30; j++) text += j == 0 ? "turn on the kitchen light" : " and turn on the kitchen light";
  Synth s(fx.lex);
  s.confusion = 0.1f;
  size_t max_nodes = 0;
  SttResult r;
  for (auto& m : toMessages(s.sentence(text))) {
    if (m.type == PhonemeFrame::EndOfUtterance) {
      r = dec.finish();
    } else {
      dec.write(m);
      max_nodes = std::max(max_nodes, dec.historyNodes());
    }
  }
  std::printf("  %zu words, at most %zu history nodes\n", r.nbest.empty() ? (size_t)0 : r.nbest[0].words.size(), max_nodes);
  CHECK(!r.nbest.empty() && r.nbest[0].words.size() >= 140);
  CHECK(max_nodes < 2048);  // beam 32
}

TEST(decoder_partial_results) {
  Fixture fx;
  auto r = fx.decode("turn on the kitchen light");
  CHECK(fx.partials.size() >= 3);
  // partials grow word by word and are prefixes of the final text
  std::string final_text = r.text();
  size_t last_len = 0;
  for (auto& p : fx.partials) {
    std::string t = p.text();
    CHECK(!p.is_final);
    CHECK(final_text.compare(0, t.size(), t) == 0);
    CHECK(t.size() >= last_len);
    last_len = t.size();
  }
  fx.dec.config().partial_results = false;
  fx.partials.clear();
  fx.decode("hello world");
  CHECK(fx.partials.empty());
}

TEST(decoder_word_list_mode) {
  Fixture fx;
  fx.lex.setUseStatic(false);
  fx.lex.addWord("go");
  fx.lex.addWord("stop");
  LanguageModel uniform;  // no sections: uniform
  uniform.setLexiconWords(fx.lex.staticWordCount());
  LanguageModelScorer us(uniform);
  fx.dec.begin(fx.lex, us);
  Synth s(fx.lex);
  auto r = fx.decode(toMessages(s.sentence("stop")));
  CHECK_STR(r.text(), "stop");
  // a word outside the list cannot be recognized
  fx.lex.setUseStatic(true);
  Synth s2(fx.lex);
  auto frames = s2.sentence("kitchen");
  fx.lex.setUseStatic(false);
  auto r2 = fx.decode(toMessages(frames));
  CHECK(std::string(r2.text()) != "kitchen");
}

TEST(decoder_hotword_boost) {
  Fixture fx;
  // "right" and "write" sound identical; without context the LM decides
  auto plain = fx.decode("right");
  std::string base = plain.text();
  CHECK(base == "right" || base == "write");
  const char* other = base == "right" ? "write" : "right";
  fx.lex.addWord(other, 5.0f);
  fx.dec.config().nbest = 5;
  auto boosted = fx.decode(toMessages(Synth(fx.lex).sentence("right")));
  CHECK_STR(boosted.text(), other);
}

TEST(decoder_nbest_and_confidence) {
  Fixture fx;
  fx.dec.config().nbest = 3;
  auto clean = fx.decode("set a timer for two minutes");
  CHECK(!clean.nbest.empty());
  CHECK(clean.nbest.size() <= 3);
  for (size_t j = 1; j < clean.nbest.size(); j++) {
    CHECK(clean.nbest[j - 1].score >= clean.nbest[j].score);
    CHECK(clean.nbest[j - 1].text != clean.nbest[j].text);
  }
  CHECK_EQ(clean.nbest[0].word_confidence.size(), clean.nbest[0].words.size());
  for (float c : clean.nbest[0].word_confidence) CHECK(c > 0 && c <= 1.0f);

  // flatter posteriors -> lower confidence
  Synth s(fx.lex);
  s.peak = 0.35f;
  auto noisy = fx.decode(toMessages(s.sentence("set a timer for two minutes")));
  std::printf("  confidence clean %.3f, noisy %.3f\n", clean.confidence(), noisy.confidence());
  CHECK(noisy.confidence() < clean.confidence());

  fx.dec.config().min_confidence = 0.999f;
  auto rejected = fx.decode(toMessages(s.sentence("set a timer for two minutes")));
  CHECK(!rejected.accepted);
}

TEST(decoder_silence_gives_empty_result) {
  Fixture fx;
  std::vector<std::vector<float>> frames(20, Synth::dist(Phonemes::kBlank, 0.97f));
  auto r = fx.decode(toMessages(frames));
  CHECK(r.nbest.empty());
  CHECK_STR(r.text(), "");
  CHECK(!r.accepted);
}

TEST(decoder_reset_message) {
  Fixture fx;
  Synth s(fx.lex);
  auto msgs = toMessages(s.sentence("turn on the fan"));
  msgs.pop_back();  // drop EOU
  for (auto& m : msgs) fx.dec.write(m);
  fx.dec.write(PhonemeFrame::reset());
  auto r = fx.decode("hello world");
  CHECK_STR(r.text(), "hello world");
}

// ---- grammar ----

TEST(grammar_parse_and_accept) {
  Lexicon lex;
  Grammar g;
  CHECK(g.compile("turn (on | off) the (kitchen | bedroom)? (light | fan);\nset a timer for (one|two|four) minutes",
                  lex));
  CHECK(g.accepts({"turn", "on", "the", "light"}));
  CHECK(g.accepts({"turn", "off", "the", "bedroom", "fan"}));
  CHECK(g.accepts({"set", "a", "timer", "for", "two", "minutes"}));
  CHECK(!g.accepts({"turn", "on", "the"}));
  CHECK(!g.accepts({"turn", "the", "light"}));
  CHECK(!g.accepts({"turn", "on", "the", "kitchen", "bedroom", "light"}));
  CHECK(g.words().count("kitchen") == 1);
  CHECK(lex.lookup("kitchen") >= 0);  // added to the overlay (letter-to-sound here)
  // nested groups and optional words
  Grammar n;
  CHECK(n.compile("(please)? (go | (turn | switch) (on | off))", lex));
  CHECK(n.accepts({"go"}));
  CHECK(n.accepts({"please", "switch", "off"}));
  CHECK(!n.accepts({"please"}));
}

TEST(grammar_syntax_errors) {
  Lexicon lex;
  Grammar g;
  CHECK(!g.compile("turn (on | off", lex));
  CHECK(!g.error().empty());
  CHECK(!g.compile("on | off", lex));
  CHECK(!g.compile("turn ()", lex));
  CHECK(!g.compile("turn on)", lex));
  CHECK(!g.compile("", lex));
}

TEST(grammar_constrained_decoding) {
  Fixture fx;
  fx.lex.setUseStatic(false);
  Grammar g;
  CHECK(g.compile("turn (on | off) the (kitchen | bedroom)? (light | fan)", fx.lex));
  fx.dec.begin(fx.lex, g);
  // all paths of the grammar decode correctly
  for (const char* t : {"turn on the light", "turn off the fan", "turn on the kitchen light",
                        "turn off the bedroom fan"}) {
    fx.lex.setUseStatic(true);
    Synth s(fx.lex);  // pronunciations from the dictionary
    auto frames = s.sentence(t);
    fx.lex.setUseStatic(false);
    auto r = fx.decode(toMessages(frames));
    CHECK_STR(r.text(), t);
  }
  // out-of-grammar speech: no complete phrase or low confidence
  fx.lex.setUseStatic(true);
  Synth s(fx.lex);
  auto frames = s.sentence("what time is it");
  fx.lex.setUseStatic(false);
  fx.dec.config().min_confidence = 0.5f;
  auto r = fx.decode(toMessages(frames));
  CHECK(!r.accepted || r.nbest.empty());
}

TEST_MAIN()
