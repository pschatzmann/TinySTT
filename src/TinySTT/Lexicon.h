#pragma once
#include <string.h>

#include <string>
#include <vector>

#include "TinySTT/LetterToSound.h"
#include "TinySTT/ModelContainer.h"

namespace tinystt {

/**
 * @brief Pronunciation lexicon as a phoneme trie, used by the decoder to
 * constrain phoneme sequences to words.
 *
 * Two parts are searched together:
 *  - a static trie built by research/build_lexicon.py from CMUdict and read
 *    zero-copy from the ModelContainer (flash friendly), and
 *  - a small dynamic overlay in RAM for user words and hotwords (addWord()).
 *
 * Word ids: static words 0 .. staticWordCount()-1 (alphabetically sorted, the
 * same ids the language model uses), overlay words follow. Overlay words can
 * be added before or after begin(); an overlay id returned before begin()
 * shifts by the static word count once the static part is loaded.
 *
 * Container sections (see docs/model-data.md):
 *   lex.nodes        Bytes, 12 B per node: first_child u32 | first_word u32 |
 *                    phone u8 | child_count u8 | word_count u8 | pad u8
 *                    (node 0 = root, children contiguous and sorted by phone)
 *   lex.node_words   U32 word ids referenced by the nodes
 *   lex.word_offsets U32 [W+1] into lex.word_text (words sorted bytewise)
 *   lex.word_text    Bytes, lower case words without separators
 *   lex.pron_offsets U32 [W+1] into lex.prons
 *   lex.prons        U8 phoneme ids (first pronunciation of every word)
 */
class Lexicon {
 public:
  static constexpr int kMaxPhones = 32;

  /// Position in the trie(s); -1 = not present in that part
  struct NodeRef {
    int32_t s = -1;  ///< static node index
    int32_t o = -1;  ///< overlay node index
    bool valid() const { return s >= 0 || o >= 0; }
    bool operator==(const NodeRef& r) const { return s == r.s && o == r.o; }
  };

  Lexicon() { clearOverlay(); }

  /// Binds the static lexicon (optional: a lexicon can be overlay only)
  bool begin(const ModelContainer& mc) {
    auto nodes = mc.find("lex.nodes");
    auto nw = mc.find("lex.node_words");
    auto wo = mc.find("lex.word_offsets");
    auto wt = mc.find("lex.word_text");
    auto po = mc.find("lex.pron_offsets");
    auto pr = mc.find("lex.prons");
    if (!nodes.valid() || !nw.u32() || !wo.u32() || !wt.valid() || !po.u32() || !pr.valid() ||
        nodes.size % kNodeSize != 0 || wo.count < 1 || po.count != wo.count) {
      LOGE("TinySTT: lexicon sections missing or invalid");
      return false;
    }
    p_nodes = nodes.data;
    node_count = nodes.size / kNodeSize;
    p_node_words = nw.u32();
    p_word_offsets = wo.u32();
    p_word_text = (const char*)wt.data;
    p_pron_offsets = po.u32();
    p_prons = pr.data;
    static_words = wo.count - 1;
    use_static = true;
    return true;
  }

  void end() {
    p_nodes = nullptr;
    node_count = static_words = 0;
    use_static = false;
    clearOverlay();
  }

  // ---------------------------------------------------------------- words

  int staticWordCount() const { return static_words; }
  int wordCount() const { return static_words + (int)overlay_words.size(); }

  /// Text of a word id ("" if invalid)
  std::string word(int id) const {
    if (id >= 0 && id < static_words) {
      uint32_t a = p_word_offsets[id], b = p_word_offsets[id + 1];
      return std::string(p_word_text + a, b - a);
    }
    int o = id - static_words;
    if (o >= 0 && o < (int)overlay_words.size()) return overlay_words[o].text;
    return "";
  }

  /// Word id of a static word (binary search), -1 if not present
  int lookupStatic(const char* text) const {
    std::string w = normalize(text);
    int lo = 0, hi = static_words - 1;
    while (lo <= hi) {
      int mid = (lo + hi) / 2;
      int c = compareStatic(mid, w);
      if (c == 0) return mid;
      if (c < 0)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    return -1;
  }

  /// Word id in the overlay or (if enabled) the static lexicon, -1 if missing
  int lookup(const char* text) const {
    std::string w = normalize(text);
    for (size_t j = 0; j < overlay_words.size(); j++)
      if (overlay_words[j].text == w) return static_words + (int)j;
    return use_static ? lookupStatic(text) : -1;
  }

  /// Pronunciation of a word id; returns the number of phonemes
  int pronunciation(int id, uint8_t* out, int max) const {
    if (id >= 0 && id < static_words) {
      uint32_t a = p_pron_offsets[id], b = p_pron_offsets[id + 1];
      int n = 0;
      for (uint32_t j = a; j < b && n < max; j++) out[n++] = p_prons[j];
      return n;
    }
    int o = id - static_words;
    if (o < 0 || o >= (int)overlay_words.size()) return 0;
    int n = 0;
    for (uint8_t p : overlay_words[o].phones)
      if (n < max) out[n++] = p;
    return n;
  }

  /// Adds a word with an explicit pronunciation to the overlay; returns its id
  int addWord(const char* text, const uint8_t* phones, int n, float boost = 0.0f) {
    if (n <= 0) return -1;
    std::string w = normalize(text);
    for (size_t j = 0; j < overlay_words.size(); j++) {
      if (overlay_words[j].text == w &&
          overlay_words[j].phones == std::vector<uint8_t>(phones, phones + n)) {
        overlay_words[j].boost = boost;
        return static_words + (int)j;
      }
    }
    int index = (int)overlay_words.size();
    overlay_words.push_back({w, std::vector<uint8_t>(phones, phones + n), boost});
    int node = 0;
    for (int j = 0; j < n; j++) node = overlayChild(node, phones[j], true);
    // store the overlay index: the id depends on the static part, which may
    // be loaded later (begin())
    overlay[node].words.push_back(index);
    return static_words + index;
  }

  /**
   * Adds a word to the overlay using the static pronunciation if known, else
   * letter-to-sound rules. Returns the new id (-1 if no pronunciation).
   */
  int addWord(const char* text, float boost = 0.0f) {
    uint8_t phones[kMaxPhones];
    int n = 0;
    int sid = lookupStatic(text);
    if (sid >= 0)
      n = pronunciation(sid, phones, kMaxPhones);
    else
      n = LetterToSound::convert(text, phones, kMaxPhones);
    return addWord(text, phones, n, boost);
  }

  /// Hotword bonus (log domain) of a word id; 0 for normal words
  float boost(int id) const {
    int o = id - static_words;
    return (o >= 0 && o < (int)overlay_words.size()) ? overlay_words[o].boost : 0.0f;
  }

  void clearOverlay() {
    overlay.clear();
    overlay.push_back(OverlayNode());
    overlay_words.clear();
  }

  /// false: only overlay words are recognized (word-list / command mode)
  void setUseStatic(bool flag) { use_static = flag && p_nodes != nullptr; }
  bool useStatic() const { return use_static; }

  // ---------------------------------------------------------------- trie

  NodeRef root() const {
    NodeRef r;
    r.s = use_static ? 0 : -1;
    r.o = overlay[0].children.empty() ? -1 : 0;
    return r;
  }

  NodeRef child(NodeRef ref, uint8_t phone) const {
    NodeRef r;
    if (ref.s >= 0) {
      uint32_t first = rd32(node(ref.s));
      int count = node(ref.s)[9];
      for (int j = 0; j < count; j++) {
        uint8_t ph = node(first + j)[8];
        if (ph == phone) {
          r.s = (int32_t)(first + j);
          break;
        }
        if (ph > phone) break;
      }
    }
    if (ref.o >= 0) r.o = const_cast<Lexicon*>(this)->overlayChild(ref.o, phone, false);
    return r;
  }

  bool hasChildren(NodeRef ref) const {
    return (ref.s >= 0 && node(ref.s)[9] > 0) || (ref.o >= 0 && !overlay[ref.o].children.empty());
  }

  /// Word ids ending at this node; returns the count (up to max)
  int wordsAt(NodeRef ref, int* out, int max) const {
    int n = 0;
    if (ref.s >= 0) {
      const uint8_t* nd = node(ref.s);
      uint32_t first = rd32(nd + 4);
      for (int j = 0; j < nd[10] && n < max; j++) out[n++] = (int)p_node_words[first + j];
    }
    if (ref.o >= 0)
      for (int index : overlay[ref.o].words)
        if (n < max) out[n++] = static_words + index;
    return n;
  }

  size_t staticNodeCount() const { return node_count; }
  size_t overlayNodeCount() const { return overlay.size(); }

  static std::string normalize(const char* text) {
    std::string w;
    for (const char* p = text; *p; p++) {
      char c = *p;
      if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
      if ((c >= 'a' && c <= 'z') || c == '\'' || (c >= '0' && c <= '9')) w += c;
    }
    return w;
  }

 protected:
  static constexpr int kNodeSize = 12;
  struct OverlayNode {
    uint8_t phone = 0;
    std::vector<int> children;
    std::vector<int> words;
  };
  struct OverlayWord {
    std::string text;
    std::vector<uint8_t> phones;
    float boost = 0.0f;
  };

  const uint8_t* p_nodes = nullptr;
  size_t node_count = 0;
  const uint32_t* p_node_words = nullptr;
  const uint32_t* p_word_offsets = nullptr;
  const char* p_word_text = nullptr;
  const uint32_t* p_pron_offsets = nullptr;
  const uint8_t* p_prons = nullptr;
  int static_words = 0;
  bool use_static = false;
  std::vector<OverlayNode> overlay;
  std::vector<OverlayWord> overlay_words;

  const uint8_t* node(uint32_t idx) const { return p_nodes + (size_t)idx * kNodeSize; }
  static uint32_t rd32(const uint8_t* p) { return ModelContainer::rd32(p); }

  int compareStatic(int id, const std::string& w) const {
    uint32_t a = p_word_offsets[id], b = p_word_offsets[id + 1];
    size_t n = b - a;
    int c = strncmp(p_word_text + a, w.c_str(), n < w.size() ? n : w.size());
    if (c != 0) return c;
    return n < w.size() ? -1 : (n > w.size() ? 1 : 0);
  }

  int overlayChild(int idx, uint8_t phone, bool create) {
    for (int c : overlay[idx].children)
      if (overlay[c].phone == phone) return c;
    if (!create) return -1;
    OverlayNode n;
    n.phone = phone;
    overlay.push_back(n);
    int id = (int)overlay.size() - 1;
    overlay[idx].children.push_back(id);
    return id;
  }
};

}  // namespace tinystt
