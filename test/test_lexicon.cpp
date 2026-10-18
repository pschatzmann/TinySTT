// Step 6 + 7: Lexicon (static trie + overlay + letter-to-sound) and the
// bigram LanguageModel, against data built by research/make_test_data.py.
#include <fstream>
#include <sstream>

#include "AudioTools.h"
#include "TestData.h"
#include "TestUtil.h"
#include "TinySTT/LanguageModel.h"
#include "TinySTT/Lexicon.h"

using namespace tinystt;

static std::vector<uint8_t> g_img = readTestFile("test_lexicon_lm.tstt");

static std::vector<uint8_t> phones(const char* arpabet) {
  std::vector<uint8_t> out;
  std::istringstream in(arpabet);
  std::string p;
  while (in >> p) out.push_back(Phonemes::id(p.c_str()));
  return out;
}

static Lexicon::NodeRef walk(const Lexicon& lex, const std::vector<uint8_t>& ph) {
  auto n = lex.root();
  for (auto p : ph) n = lex.child(n, p);
  return n;
}

static std::vector<std::string> wordsFor(const Lexicon& lex, const char* arpabet) {
  std::vector<std::string> out;
  auto n = walk(lex, phones(arpabet));
  int ids[16];
  int k = lex.wordsAt(n, ids, 16);
  for (int j = 0; j < k; j++) out.push_back(lex.word(ids[j]));
  std::sort(out.begin(), out.end());
  return out;
}

static std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (auto& x : v) s += (s.empty() ? "" : ",") + x;
  return s;
}

TEST(lexicon_static_lookup) {
  ModelContainer mc;
  CHECK_EQ(mc.begin(g_img.data(), g_img.size()), ModelContainer::OK);
  Lexicon lex;
  CHECK(lex.begin(mc));
  CHECK_EQ(lex.staticWordCount(), 33);
  int id = lex.lookup("Kitchen");
  CHECK(id >= 0);
  CHECK_STR(lex.word(id), "kitchen");
  uint8_t p[16];
  int n = lex.pronunciation(id, p, 16);
  CHECK_EQ(n, 5);
  CHECK_EQ(p[0], Phonemes::id("K"));
  CHECK_EQ(lex.lookup("zebra"), -1);
  // ids are alphabetical
  for (int j = 1; j < lex.staticWordCount(); j++) CHECK(lex.word(j - 1) < lex.word(j));
}

TEST(lexicon_trie_homophones_and_variants) {
  ModelContainer mc;
  mc.begin(g_img.data(), g_img.size());
  Lexicon lex;
  lex.begin(mc);
  CHECK_STR(join(wordsFor(lex, "T UW")), "to,too,two");
  CHECK_STR(join(wordsFor(lex, "R AY T")), "right,write");
  CHECK_STR(join(wordsFor(lex, "F AO R")), "for,four");
  CHECK_STR(join(wordsFor(lex, "HH EH L OW")), "hello");  // second variant
  CHECK_STR(join(wordsFor(lex, "HH AH L OW")), "hello");
  CHECK_STR(join(wordsFor(lex, "DH IY")), "the");
  // prefix of "lights": "light" ends here and the node still has children
  auto n = walk(lex, phones("L AY T"));
  CHECK(lex.hasChildren(n));
  CHECK_STR(join(wordsFor(lex, "L AY T S")), "lights");
  // dead ends
  CHECK(!walk(lex, phones("ZH ZH")).valid());
  CHECK(wordsFor(lex, "T").empty());
}

TEST(lexicon_overlay_and_word_list_mode) {
  ModelContainer mc;
  mc.begin(g_img.data(), g_img.size());
  Lexicon lex;
  lex.begin(mc);
  int sid = lex.lookup("light");
  // known word: overlay copy keeps the dictionary pronunciation
  int oid = lex.addWord("light", 2.0f);
  CHECK(oid >= lex.staticWordCount());
  CHECK_NEAR(lex.boost(oid), 2.0, 0);
  CHECK_NEAR(lex.boost(sid), 0.0, 0);
  // unknown word: letter-to-sound
  int name = lex.addWord("Schatzmann");
  CHECK(name > oid);
  uint8_t p[32];
  int n = lex.pronunciation(name, p, 32);
  CHECK(n >= 5);
  CHECK_EQ(p[0], Phonemes::id("SH"));
  // the overlay word is reachable through the trie
  std::vector<uint8_t> ph(p, p + n);
  int ids[8];
  CHECK_EQ(lex.wordsAt(walk(lex, ph), ids, 8), 1);
  CHECK_EQ(ids[0], name);
  // word-list mode: only overlay words
  lex.setUseStatic(false);
  CHECK_STR(join(wordsFor(lex, "T UW")), "");
  CHECK_STR(join(wordsFor(lex, "L AY T")), "light");
  CHECK_EQ(lex.lookup("kitchen"), -1);
  CHECK_EQ(lex.lookup("light"), oid);
  // adding the same word twice returns the same id
  CHECK_EQ(lex.addWord("light", 3.0f), oid);
  CHECK_NEAR(lex.boost(oid), 3.0, 0);
}

TEST(lexicon_overlay_only_without_container) {
  Lexicon lex;
  CHECK_EQ(lex.staticWordCount(), 0);
  CHECK(!lex.root().valid());
  auto ph = phones("G OW");
  int id = lex.addWord("go", ph.data(), (int)ph.size());
  CHECK_EQ(id, 0);
  CHECK(lex.root().valid());
  CHECK_STR(join(wordsFor(lex, "G OW")), "go");
}

TEST(letter_to_sound_examples) {
  auto lts = [](const char* w) {
    uint8_t p[32];
    int n = LetterToSound::convert(w, p, 32);
    std::string s;
    for (int j = 0; j < n; j++) s += std::string(s.empty() ? "" : " ") + Phonemes::name(p[j]);
    return s;
  };
  CHECK_STR(lts("ship"), "SH IH P");
  CHECK_STR(lts("check"), "CH EH K");
  CHECK_STR(lts("night"), "N AY T");
  CHECK_STR(lts("make"), "M AE K");  // silent final e
  CHECK_STR(lts("station"), "S T AE SH AH N");
  CHECK_STR(lts("Tom!"), "T AA M");
  CHECK_STR(lts(""), "");
}

TEST(lm_matches_python_reference) {
  ModelContainer mc;
  mc.begin(g_img.data(), g_img.size());
  Lexicon lex;
  lex.begin(mc);
  LanguageModel lm;
  CHECK(lm.begin(mc, lex.staticWordCount()));
  CHECK(!lm.isUniform());
  std::ifstream ref(TINYSTT_TEST_DATA "/test_lm_reference.txt");
  CHECK(ref.good());
  std::string h, w;
  float expected;
  int lines = 0;
  auto id = [&](const std::string& s) {
    if (s == "<s>") return lm.bos();
    if (s == "</s>") return lm.eos();
    if (s == "<unk>") return lm.unk();
    return lex.lookup(s.c_str());
  };
  while (ref >> h >> w >> expected) {
    CHECK_NEAR(lm.logProb(id(h), id(w)), expected, 1e-4);
    lines++;
  }
  CHECK_EQ(lines, 13);
  // bigram is preferred over backoff, and sums to ~1 over the vocabulary
  CHECK(lm.logProb(id("turn"), id("on")) > lm.logProb(id("turn"), id("kitchen")));
  for (const char* hist : {"<s>", "turn", "the", "zzz_unknown"}) {
    int hid = std::string(hist) == "zzz_unknown" ? lm.unk() : id(hist);
    double sum = 0;
    for (int v = 0; v < (int)lm.vocabSize(); v++)
      if (v != lm.bos()) sum += std::exp(lm.logProb(hid, v));
    CHECK_NEAR(sum, 1.0, 0.02);  // int16 quantization of the log values
  }
  // overlay ids are scored as <unk>
  CHECK_NEAR(lm.logProb(id("the"), lex.staticWordCount() + 50), lm.logProb(id("the"), lm.unk()), 0);
}

TEST(lm_uniform_without_sections) {
  ModelContainerWriter w;
  w.addString("info", "empty");
  auto img = w.build();
  ModelContainer mc;
  mc.begin(img.data(), img.size());
  LanguageModel lm;
  CHECK(!lm.begin(mc, 10));
  CHECK(lm.isUniform());
  CHECK_NEAR(lm.logProb(lm.bos(), 3), -std::log(12.0), 1e-6);
}

TEST(lm_rejects_mismatched_lexicon) {
  ModelContainer mc;
  mc.begin(g_img.data(), g_img.size());
  LanguageModel lm;
  CHECK(!lm.begin(mc, 20));
  CHECK(lm.isUniform());
}

TEST_MAIN()
