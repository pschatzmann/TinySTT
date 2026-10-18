#pragma once
#include <math.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "TinySTT/Lexicon.h"
#include "TinySTT/WordScorer.h"

namespace tinystt {

/**
 * @brief Command grammar: constrains the decoder to word sequences described
 * by a small expression language, e.g.
 *
 *     turn (on | off) the (kitchen | bedroom)? (light | fan);
 *     set a timer for (one | two | four) minutes
 *
 *  - words are matched case-insensitively against the lexicon;
 *  - ( a | b ) alternatives, ? makes the preceding word or group optional;
 *  - ';' or a new line separates alternative top-level phrases.
 *
 * compile() builds an NFA and determinizes it (subset construction) into a
 * word-level DFA; next() scores a word -log(branching) so the grammar acts
 * as a uniform language model over the allowed phrases. Words are added to
 * the lexicon overlay (dictionary pronunciation or letter-to-sound), so the
 * grammar also works with Lexicon::setUseStatic(false).
 */
class Grammar : public WordScorer {
 public:
  /// Compiles the grammar; returns false with error() on a syntax error
  bool compile(const char* source, Lexicon& lexicon, bool addWords = true) {
    p_lex = &lexicon;
    error_msg.clear();
    nfa.assign(1, NfaState());  // state 0 = start
    src = source;
    pos = 0;
    int end_state = newState();
    // top level: alternatives separated by ';' or newline
    while (true) {
      skipSpace(false);
      if (peek() == 0) break;
      if (peek() == ';' || peek() == '\n') {
        pos++;
        continue;
      }
      int s = newState(), e = -1;
      addEps(0, s);
      if (!parseSequence(s, e, true)) return false;
      addEps(e, end_state);
    }
    nfa[end_state].final = true;
    if (nfa[0].eps.empty()) return fail("empty grammar");
    determinize();
    if (addWords) {
      for (auto& w : vocabulary) {
        if (lexicon.addWord(w.c_str()) < 0) return fail("no pronunciation for '" + w + "'");
      }
    }
    return true;
  }

  /// Compiles the grammar stored as string section `name` in the container
  bool compile(const ModelContainer& mc, const char* name, Lexicon& lexicon, bool addWords = true) {
    auto s = mc.find(name);
    if (!s.valid()) return fail(std::string("missing grammar section ") + name);
    std::string text((const char*)s.data, strnlen((const char*)s.data, s.size));
    return compile(text.c_str(), lexicon, addWords);
  }

  const std::string& error() const { return error_msg; }
  const std::set<std::string>& words() const { return vocabulary; }
  size_t stateCount() const { return dfa.size(); }

  /// True if the word sequence is a complete phrase of the grammar
  bool accepts(const std::vector<std::string>& phrase) const {
    if (dfa.empty()) return false;
    int s = 0;
    for (auto& w : phrase) {
      auto it = dfa[s].next.find(Lexicon::normalize(w.c_str()));
      if (it == dfa[s].next.end()) return false;
      s = it->second;
    }
    return dfa[s].final;
  }

  // WordScorer
  int startState() const override { return 0; }
  bool next(int state, int word, int& next_state, float& score) const override {
    if (state < 0 || state >= (int)dfa.size() || p_lex == nullptr) return false;
    auto it = dfa[state].next.find(p_lex->word(word));
    if (it == dfa[state].next.end()) return false;
    next_state = it->second;
    score = -logf((float)dfa[state].next.size() + (dfa[state].final ? 1.0f : 0.0f));
    return true;
  }
  bool end(int state, float& score) const override {
    if (state < 0 || state >= (int)dfa.size() || !dfa[state].final) return false;
    score = -logf((float)dfa[state].next.size() + 1.0f);
    return true;
  }

 protected:
  struct NfaState {
    std::vector<int> eps;
    std::vector<std::pair<std::string, int>> arcs;
    bool final = false;
  };
  struct DfaState {
    std::map<std::string, int> next;
    bool final = false;
  };

  const Lexicon* p_lex = nullptr;
  std::vector<NfaState> nfa;
  std::vector<DfaState> dfa;
  std::set<std::string> vocabulary;
  std::string error_msg;
  const char* src = "";
  size_t pos = 0;

  bool fail(const std::string& msg) {
    error_msg = msg;
    dfa.clear();
    return false;
  }
  int newState() {
    nfa.push_back(NfaState());
    return (int)nfa.size() - 1;
  }
  void addEps(int a, int b) { nfa[a].eps.push_back(b); }
  char peek() const { return src[pos]; }
  void skipSpace(bool newlines) {
    while (src[pos] == ' ' || src[pos] == '\t' || src[pos] == '\r' || (newlines && src[pos] == '\n')) pos++;
  }

  /// sequence := item* ; stops at ')' '|' ';' newline (top level) or end
  bool parseSequence(int start, int& end, bool top) {
    int cur = start;
    while (true) {
      skipSpace(!top);
      char c = peek();
      if (c == 0 || c == ')' || c == '|' || ((c == ';' || c == '\n') && top)) break;
      if ((c == ';' || c == '\n') && !top) return fail("';' inside a group");
      int s = cur, e = -1;
      if (!parseItem(s, e)) return false;
      cur = e;
    }
    end = cur;
    if (top && peek() == '|') return fail("'|' outside of a group - use ( a | b )");
    if (top && peek() == ')') return fail("unbalanced ')'");
    return true;
  }

  /// item := (word | '(' alternatives ')') '?'?
  bool parseItem(int start, int& end) {
    int s = newState(), e = newState();
    addEps(start, s);
    if (peek() == '(') {
      pos++;
      bool any = false;
      while (true) {
        int as = newState(), ae = -1;
        addEps(s, as);
        if (!parseSequence(as, ae, false)) return false;
        if (ae == as) return fail("empty alternative");
        addEps(ae, e);
        any = true;
        skipSpace(true);
        if (peek() == '|') {
          pos++;
          continue;
        }
        if (peek() == ')') {
          pos++;
          break;
        }
        return fail("missing ')'");
      }
      if (!any) return fail("empty group");
    } else {
      std::string w;
      while (isWordChar(peek())) w += src[pos++];
      w = Lexicon::normalize(w.c_str());
      if (w.empty()) return fail(std::string("unexpected character '") + peek() + "'");
      nfa[s].arcs.push_back({w, e});
      vocabulary.insert(w);
    }
    skipSpace(false);
    if (peek() == '?') {
      pos++;
      addEps(s, e);
    }
    end = e;
    return true;
  }

  static bool isWordChar(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '\'' || c == '-';
  }

  std::vector<int> closure(std::vector<int> states) const {
    std::vector<int> stack = states;
    std::set<int> seen(states.begin(), states.end());
    while (!stack.empty()) {
      int s = stack.back();
      stack.pop_back();
      for (int t : nfa[s].eps)
        if (seen.insert(t).second) stack.push_back(t);
    }
    return std::vector<int>(seen.begin(), seen.end());
  }

  void determinize() {
    dfa.clear();
    std::map<std::vector<int>, int> index;
    std::vector<std::vector<int>> sets;
    auto get = [&](const std::vector<int>& set) {
      auto it = index.find(set);
      if (it != index.end()) return it->second;
      int id = (int)sets.size();
      index[set] = id;
      sets.push_back(set);
      dfa.push_back(DfaState());
      for (int s : set)
        if (nfa[s].final) dfa[id].final = true;
      return id;
    };
    get(closure({0}));
    for (size_t i = 0; i < sets.size(); i++) {
      std::map<std::string, std::vector<int>> moves;
      for (int s : sets[i])
        for (auto& a : nfa[s].arcs) moves[a.first].push_back(a.second);
      for (auto& m : moves) {
        int target = get(closure(m.second));
        dfa[i].next[m.first] = target;
      }
    }
  }
};

}  // namespace tinystt
