#pragma once
#include <stdint.h>
#include <string.h>

namespace tinystt {

/**
 * @brief Phoneme inventory shared by both stages: CTC blank, silence and the
 * 39 stress-less ARPAbet phonemes of the CMU Pronouncing Dictionary (the same
 * base set TinyTTS uses, see TinyTTS docs/phonemes.md).
 *
 * Ids are stable: they are the acoustic model's output classes, the ids in
 * PhonemeFrame and on the wire, and the ids in the lexicon.
 */
struct Phonemes {
  static constexpr uint8_t kBlank = 0;
  static constexpr uint8_t kSilence = 1;
  static constexpr uint8_t kFirstPhone = 2;
  static constexpr int kCount = 41;  ///< number of model output classes
  static constexpr uint8_t kInvalid = 0xFF;

  /// Symbol for an id ("<b>" for blank, "sil" for silence), "?" if invalid
  static const char* name(uint8_t id) {
    return id < kCount ? table()[id] : "?";
  }

  /// Id for an ARPAbet symbol; case-insensitive, a trailing stress digit
  /// (AH0, AH1, AH2) is ignored. Returns kInvalid if unknown.
  static uint8_t id(const char* symbol) {
    if (symbol == nullptr) return kInvalid;
    char buf[8];
    size_t n = 0;
    for (; symbol[n] != 0 && n < sizeof(buf) - 1; n++) {
      char c = symbol[n];
      if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
      buf[n] = c;
    }
    if (symbol[n] != 0) return kInvalid;
    if (n > 0 && buf[n - 1] >= '0' && buf[n - 1] <= '2') n--;
    buf[n] = 0;
    if (strcmp(buf, "SIL") == 0) return kSilence;
    if (strcmp(buf, "<B>") == 0) return kBlank;
    for (int j = kFirstPhone; j < kCount; j++) {
      if (strcmp(buf, table()[j]) == 0) return (uint8_t)j;
    }
    return kInvalid;
  }

  static bool isVowel(uint8_t id) {
    switch (id) {
      case 2: case 3: case 4: case 5: case 6: case 7:   // AA AE AH AO AW AY
      case 12: case 13: case 14:                        // EH ER EY
      case 18: case 19:                                 // IH IY
      case 26: case 27:                                 // OW OY
      case 34: case 35:                                 // UH UW
        return true;
      default:
        return false;
    }
  }

 protected:
  static const char* const* table() {
    static const char* const t[kCount] = {
        "<b>", "sil", "AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",
        "DH",  "EH",  "ER", "EY", "F",  "G",  "HH", "IH", "IY", "JH", "K",
        "L",   "M",   "N",  "NG", "OW", "OY", "P",  "R",  "S",  "SH", "T",
        "TH",  "UH",  "UW", "V",  "W",  "Y",  "Z",  "ZH"};
    return t;
  }
};

}  // namespace tinystt
