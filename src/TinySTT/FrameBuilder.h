#pragma once
#include <math.h>

#include "TinySTT/PhonemeQueue.h"

namespace tinystt {

/**
 * @brief Turns acoustic model outputs (log-probabilities per 30 ms frame) into
 * PhonemeFrames and writes them to the phoneme queue:
 *  - keeps the top-k classes per frame (k <= PhonemeFrame::kMaxTopK);
 *  - blank skipping: consecutive frames with P(blank) > threshold are merged
 *    into one BlankRun frame (threshold >= 1 disables skipping). The decoder
 *    treats a run as `run` blank steps with probability ~1, which is the
 *    CTC-consistent reading of near-certain blanks.
 */
class FrameBuilder {
 public:
  explicit FrameBuilder(PhonemeQueueWriter& writer) : p_writer(&writer) {}

  void setTopK(int k) {
    top_k = k < 1 ? 1 : (k > PhonemeFrame::kMaxTopK ? PhonemeFrame::kMaxTopK : k);
  }
  int topK() const { return top_k; }

  /// P(blank) above which a frame is merged into a BlankRun (default 0.95)
  void setBlankThreshold(float p) {
    blank_threshold = p;
    log_blank_threshold = p >= 1.0f ? 1.0f : logf(p);
  }
  float blankThreshold() const { return blank_threshold; }

  /// One model output frame with `n` log-probabilities
  void write(const float* log_probs, int n) {
    uint32_t idx = frame_index++;
    if (log_blank_threshold <= 0.0f && log_probs[Phonemes::kBlank] > log_blank_threshold) {
      if (run_length == 0) run_start = idx;
      run_length++;
      if (run_length == 0xFFFF) flushRun();
      return;
    }
    flushRun();
    PhonemeFrame f;
    f.index = idx;
    f.k = (uint8_t)(top_k < n ? top_k : n);
    // partial selection sort of the k best classes
    float best[PhonemeFrame::kMaxTopK];
    for (int j = 0; j < f.k; j++) best[j] = -INFINITY;
    for (int c = 0; c < n; c++) {
      float v = log_probs[c];
      if (v <= best[f.k - 1]) continue;
      int pos = f.k - 1;
      while (pos > 0 && best[pos - 1] < v) {
        best[pos] = best[pos - 1];
        f.id[pos] = f.id[pos - 1];
        pos--;
      }
      best[pos] = v;
      f.id[pos] = (uint8_t)c;
    }
    for (int j = 0; j < f.k; j++) f.logprob[j] = PhonemeFrame::quantize(best[j]);
    p_writer->write(f);
    emitted++;
  }

  /// Flushes a pending blank run and sends EndOfUtterance
  void endOfUtterance() {
    flushRun();
    p_writer->write(PhonemeFrame::endOfUtterance(frame_index));
    emitted++;
  }

  /// Frame numbering restarts (call together with AcousticModel::reset())
  void reset() {
    run_length = 0;
    frame_index = 0;
  }

  uint32_t frameIndex() const { return frame_index; }
  /// Queue messages produced (Frame + BlankRun + EndOfUtterance)
  uint32_t emittedMessages() const { return emitted; }

 protected:
  PhonemeQueueWriter* p_writer;
  int top_k = 4;
  float blank_threshold = 0.95f;
  float log_blank_threshold = logf(0.95f);
  uint32_t frame_index = 0;
  uint32_t run_start = 0;
  uint32_t run_length = 0;
  uint32_t emitted = 0;

  void flushRun() {
    if (run_length == 0) return;
    p_writer->write(PhonemeFrame::blankRun(run_start, (uint16_t)run_length));
    emitted++;
    run_length = 0;
  }
};

}  // namespace tinystt
