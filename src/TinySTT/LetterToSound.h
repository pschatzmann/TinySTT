#pragma once
#include <ctype.h>
#include <string.h>

#include "TinySTT/Phonemes.h"

namespace tinystt {

/**
 * @brief Small rule-based English letter-to-sound conversion, used for words
 * that are not in the lexicon (user words, hotwords, names). Greedy longest
 * match over grapheme rules; good enough to give the decoder a plausible
 * pronunciation, not a replacement for a dictionary.
 */
class LetterToSound {
 public:
  /// Writes phoneme ids for `word` to out (max entries); returns the count
  static int convert(const char* word, uint8_t* out, int max) {
    char w[64];
    int len = 0;
    for (const char* p = word; *p && len < (int)sizeof(w) - 1; p++) {
      char c = (char)tolower((unsigned char)*p);
      if (c >= 'a' && c <= 'z') w[len++] = c;
    }
    w[len] = 0;
    int n = 0;
    int pos = 0;
    while (pos < len && n < max) {
      const Rule* best = nullptr;
      size_t best_len = 0;
      for (const Rule* r = rules(); r->graph != nullptr; r++) {
        size_t gl = strlen(r->graph);
        if (gl <= best_len || pos + (int)gl > len) continue;
        if (strncmp(w + pos, r->graph, gl) != 0) continue;
        if (r->final_only && pos + (int)gl != len) continue;
        best = r;
        best_len = gl;
      }
      if (best == nullptr) {  // unknown letter: skip
        pos++;
        continue;
      }
      // silent final 'e' after a consonant (make, note)
      if (!(best->silent_final_e && pos == len - 1 && pos > 0)) {
        for (int j = 0; j < 3 && best->phones[j] != nullptr && n < max; j++) {
          uint8_t id = Phonemes::id(best->phones[j]);
          if (id != Phonemes::kInvalid) out[n++] = id;
        }
      }
      pos += (int)best_len;
    }
    // collapse immediate duplicates (e.g. "ss" handled as S S -> S)
    int m = 0;
    for (int j = 0; j < n; j++)
      if (m == 0 || out[m - 1] != out[j]) out[m++] = out[j];
    return m;
  }

 protected:
  struct Rule {
    const char* graph;
    const char* phones[3];
    bool final_only;
    bool silent_final_e;
  };

  static const Rule* rules() {
    static const Rule r[] = {
        {"tion", {"SH", "AH", "N"}, false, false},
        {"sion", {"ZH", "AH", "N"}, false, false},
        {"ough", {"AO", nullptr, nullptr}, false, false},
        {"igh", {"AY", nullptr, nullptr}, false, false},
        {"tch", {"CH", nullptr, nullptr}, false, false},
        {"sch", {"SH", nullptr, nullptr}, false, false},
        {"ing", {"IH", "NG", nullptr}, true, false},
        {"ck", {"K", nullptr, nullptr}, false, false},
        {"ch", {"CH", nullptr, nullptr}, false, false},
        {"sh", {"SH", nullptr, nullptr}, false, false},
        {"th", {"TH", nullptr, nullptr}, false, false},
        {"ph", {"F", nullptr, nullptr}, false, false},
        {"wh", {"W", nullptr, nullptr}, false, false},
        {"qu", {"K", "W", nullptr}, false, false},
        {"ng", {"NG", nullptr, nullptr}, false, false},
        {"kn", {"N", nullptr, nullptr}, false, false},
        {"wr", {"R", nullptr, nullptr}, false, false},
        {"ee", {"IY", nullptr, nullptr}, false, false},
        {"ea", {"IY", nullptr, nullptr}, false, false},
        {"oo", {"UW", nullptr, nullptr}, false, false},
        {"ou", {"AW", nullptr, nullptr}, false, false},
        {"ow", {"OW", nullptr, nullptr}, false, false},
        {"oi", {"OY", nullptr, nullptr}, false, false},
        {"oy", {"OY", nullptr, nullptr}, false, false},
        {"ai", {"EY", nullptr, nullptr}, false, false},
        {"ay", {"EY", nullptr, nullptr}, false, false},
        {"au", {"AO", nullptr, nullptr}, false, false},
        {"aw", {"AO", nullptr, nullptr}, false, false},
        {"er", {"ER", nullptr, nullptr}, false, false},
        {"ir", {"ER", nullptr, nullptr}, false, false},
        {"ur", {"ER", nullptr, nullptr}, false, false},
        {"ar", {"AA", "R", nullptr}, false, false},
        {"or", {"AO", "R", nullptr}, false, false},
        {"tz", {"T", "S", nullptr}, false, false},
        {"a", {"AE", nullptr, nullptr}, false, false},
        {"b", {"B", nullptr, nullptr}, false, false},
        {"c", {"K", nullptr, nullptr}, false, false},
        {"d", {"D", nullptr, nullptr}, false, false},
        {"e", {"EH", nullptr, nullptr}, false, true},
        {"f", {"F", nullptr, nullptr}, false, false},
        {"g", {"G", nullptr, nullptr}, false, false},
        {"h", {"HH", nullptr, nullptr}, false, false},
        {"i", {"IH", nullptr, nullptr}, false, false},
        {"j", {"JH", nullptr, nullptr}, false, false},
        {"k", {"K", nullptr, nullptr}, false, false},
        {"l", {"L", nullptr, nullptr}, false, false},
        {"m", {"M", nullptr, nullptr}, false, false},
        {"n", {"N", nullptr, nullptr}, false, false},
        {"o", {"AA", nullptr, nullptr}, false, false},
        {"p", {"P", nullptr, nullptr}, false, false},
        {"r", {"R", nullptr, nullptr}, false, false},
        {"s", {"S", nullptr, nullptr}, false, false},
        {"t", {"T", nullptr, nullptr}, false, false},
        {"u", {"AH", nullptr, nullptr}, false, false},
        {"v", {"V", nullptr, nullptr}, false, false},
        {"w", {"W", nullptr, nullptr}, false, false},
        {"x", {"K", "S", nullptr}, false, false},
        {"y", {"IY", nullptr, nullptr}, true, false},
        {"y", {"Y", nullptr, nullptr}, false, false},
        {"z", {"Z", nullptr, nullptr}, false, false},
        {nullptr, {nullptr, nullptr, nullptr}, false, false},
    };
    return r;
  }
};

}  // namespace tinystt
