#pragma once
#include "AudioTools.h"

namespace tinystt {

/**
 * @brief Energy based endpointing on top of audio-tools' VolumeMeter. Fed with
 * one 10 ms block at a time; reports speech start after `start_ms` above
 * `start_db` and speech end after `hangover_ms` below `stop_db`. An utterance
 * longer than `max_utterance_ms` ends at its next short pause
 * (`split_pause_ms`), so it is not cut in the middle of a word; only at
 * `hard_max_utterance_ms` it is cut unconditionally.
 */
class VAD {
 public:
  enum Event { None, SpeechStart, SpeechEnd };

  struct Config {
    float start_db = -35.0f;      ///< max amplitude (dBFS) that counts as speech
    float stop_db = -40.0f;       ///< below this counts as silence
    uint32_t start_ms = 30;       ///< speech needed to start an utterance
    uint32_t hangover_ms = 600;   ///< silence needed to end it
    uint32_t max_utterance_ms = 15000;       ///< then end at the next short pause
    uint32_t split_pause_ms = 60;            ///< pause that ends a long utterance
    uint32_t hard_max_utterance_ms = 30000;  ///< cut unconditionally
  };

  Config& config() { return cfg; }

  void begin(int sampleRate = 16000) {
    audio_tools::AudioInfo info(sampleRate, 1, 16);
    meter.begin(info);
    reset();
  }

  void reset() {
    in_speech = false;
    above_ms = below_ms = speech_ms = 0;
  }

  /// One block of 16 bit mono samples
  Event update(const int16_t* samples, int count, int sampleRate = 16000) {
    meter.clear();
    meter.write((const uint8_t*)samples, count * sizeof(int16_t));
    last_db = meter.volumeDB();
    uint32_t ms = (uint32_t)count * 1000 / sampleRate;
    if (!in_speech) {
      above_ms = last_db > cfg.start_db ? above_ms + ms : 0;
      if (above_ms >= cfg.start_ms) {
        in_speech = true;
        below_ms = 0;
        speech_ms = above_ms;
        return SpeechStart;
      }
      return None;
    }
    speech_ms += ms;
    below_ms = last_db < cfg.stop_db ? below_ms + ms : 0;
    bool long_pause = speech_ms >= cfg.max_utterance_ms && below_ms >= cfg.split_pause_ms;
    if (below_ms >= cfg.hangover_ms || long_pause || speech_ms >= cfg.hard_max_utterance_ms) {
      in_speech = false;
      above_ms = 0;
      return SpeechEnd;
    }
    return None;
  }

  bool isSpeech() const { return in_speech; }
  float lastDB() const { return last_db; }

 protected:
  Config cfg;
  audio_tools::VolumeMeter meter;
  bool in_speech = false;
  uint32_t above_ms = 0, below_ms = 0, speech_ms = 0;
  float last_db = -1000.0f;
};

}  // namespace tinystt
