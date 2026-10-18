#pragma once
#include "AudioTools.h"
#include "TinySTT/PhonemeProtocol.h"

namespace tinystt {

/**
 * @brief Prints the phonemes of step 1 (SpeechToPhonemes) in a human readable
 * form. It implements the phoneme queue interface, so it is passed to
 * SpeechToPhonemes (or StreamToQueue) in place of a queue and prints every
 * frame as it arrives:
 *
 *   PhonemeOutput phonemes;                       // prints to Serial
 *   SpeechToPhonemes speechToPhonemes(fft, phonemes);
 *
 * Modes:
 *  - Phonemes (default): the most likely phoneme sequence of each utterance
 *    on one line, e.g. "T ER N AA N DH AH L AY T" (greedy CTC: blanks and
 *    silence dropped, repeats collapsed).
 *  - Frames: every frame with its top-k phonemes and log-probabilities, e.g.
 *    "#12 HH:-0.25 AH:-1.75" (see docs/protocol.md).
 */
class PhonemeOutput : public audio_tools::BaseBuffer<PhonemeFrame> {
 public:
  enum Mode { Phonemes, Frames };

  explicit PhonemeOutput(Print& out = Serial, Mode mode = Phonemes) : p_out(&out), out_mode(mode) {}

  void setOutput(Print& out) { p_out = &out; }
  void setMode(Mode mode) {
    out_mode = mode;
    endLine();
  }
  Mode mode() const { return out_mode; }

  /// Prints one frame; always accepted (never blocks the producer)
  bool write(PhonemeFrame frame) override {
    if (out_mode == Frames) {
      PhonemeProtocol::printText(*p_out, frame);
      return true;
    }
    switch (frame.type) {
      case PhonemeFrame::Frame: {
        uint8_t best = frame.k > 0 ? frame.id[0] : Phonemes::kBlank;
        bool is_phone = best != Phonemes::kBlank && best != Phonemes::kSilence;
        if (is_phone && best != last_id) {
          if (count > 0) p_out->print(' ');
          p_out->print(Phonemes::name(best));
          count++;
        }
        last_id = best;
        break;
      }
      case PhonemeFrame::BlankRun:
        last_id = Phonemes::kBlank;  // a blank separates repeated phonemes
        break;
      case PhonemeFrame::EndOfUtterance:
      case PhonemeFrame::Reset:
        endLine();
        break;
    }
    return true;
  }

  // nothing is stored: the remaining queue interface is empty
  bool read(PhonemeFrame&) override { return false; }
  bool peek(PhonemeFrame&) override { return false; }
  void reset() override { endLine(); }
  int available() override { return 0; }
  int availableForWrite() override { return 1024; }
  PhonemeFrame* address() override { return nullptr; }
  size_t size() override { return 0; }

 protected:
  Print* p_out;
  Mode out_mode;
  uint8_t last_id = Phonemes::kBlank;
  int count = 0;

  /// finishes the current utterance line (if anything was printed)
  void endLine() {
    if (count > 0) p_out->println();
    count = 0;
    last_id = Phonemes::kBlank;
  }
};

}  // namespace tinystt
