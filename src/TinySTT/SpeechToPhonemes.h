#pragma once
#include <vector>

#include "AudioTools.h"
#include "TinySTT/AcousticModel.h"
#include "TinySTT/FeatureExtractor.h"
#include "TinySTT/FrameBuilder.h"
#include "TinySTT/VAD.h"

namespace tinystt {

/**
 * @brief Stage 1: audio -> phonemes. An AudioOutput sink for 16 kHz mono int16
 * PCM; it is the single producer of the phoneme queue.
 *
 *   PCM -> 10 ms blocks -> VAD (VolumeMeter) + FeatureExtractor (injected FFT)
 *       -> AcousticModel -> FrameBuilder (top-k, blank runs) -> queue
 *
 * While the VAD reports silence only features are computed; the last
 * `preroll` feature frames are replayed into the model when speech starts.
 * With the VAD disabled the model runs continuously and utterances are ended
 * with endOfUtterance().
 */
class SpeechToPhonemes : public audio_tools::AudioOutput {
 public:
  SpeechToPhonemes(audio_tools::AudioFFTBase& fft, audio_tools::BaseBuffer<PhonemeFrame>& queue)
      : features(fft), writer(queue), builder(writer) {}

  /// Binds the acoustic model (and feature normalization) from the container
  bool begin(const ModelContainer& mc, AcousticModel::Precision precision = AcousticModel::Precision::Auto) {
    // everything else first: the acoustic model's weight placement (last)
    // must see the internal RAM that is really left (internalRamReserve())
    auto mean = mc.find("features.mean");
    auto inv_std = mc.find("features.inv_std");
    if (mean.f32() && inv_std.f32() && mean.count == FeatureExtractor::kMelBins &&
        inv_std.count == FeatureExtractor::kMelBins)
      features.setNormalization(mean.f32(), inv_std.f32());
    else
      features.setNormalization(nullptr, nullptr);
    features.setFrameCallback(onFeatures, this);
    if (!features.begin()) return false;
    vad.begin(FeatureExtractor::kSampleRate);
    preroll_buf.assign((size_t)preroll_frames * FeatureExtractor::kMelBins, 0.0f);
    preroll_count = preroll_pos = 0;
    block_fill = 0;
    in_utterance = !vad_enabled;
    if (!model.begin(mc, precision)) return false;
    builder.reset();
    audio_tools::AudioOutput::setAudioInfo(audio_tools::AudioInfo(FeatureExtractor::kSampleRate, 1, 16));
    return true;
  }


  void end() override { features.end(); }

  /// 16 kHz mono int16 PCM
  size_t write(const uint8_t* data, size_t len) override {
    size_t pos = 0;
    while (pos < len) {
      size_t n = kBlockBytes - block_fill;
      if (n > len - pos) n = len - pos;
      memcpy((uint8_t*)block + block_fill, data + pos, n);
      block_fill += n;
      pos += n;
      if (block_fill == kBlockBytes) {
        processBlock();
        block_fill = 0;
      }
    }
    return len;
  }

  /// Ends the current utterance (flushes blank runs, sends EndOfUtterance)
  void endOfUtterance() {
    if (!in_utterance && vad_enabled) return;
    emitFrames(model.flush());  // incomplete batch
    builder.endOfUtterance();
    model.reset();
    builder.reset();
    in_utterance = !vad_enabled;
  }

  /// false: no endpointing, call endOfUtterance() yourself
  void setVADEnabled(bool flag) {
    vad_enabled = flag;
    in_utterance = !flag;
  }
  VAD::Config& vadConfig() { return vad.config(); }
  bool isSpeech() const { return in_utterance; }

  /// Feature frames (10 ms) replayed when speech starts (default 30)
  void setPreroll(int frames) { preroll_frames = frames < 0 ? 0 : frames; }
  void setTopK(int k) { builder.setTopK(k); }
  /// Output frames computed together, see AcousticModel::setBatchFrames()
  /// (call before begin()). 4 is a good choice on ESP32 (+90 ms latency).
  void setBatchFrames(int n) { model.setBatchFrames(n); }
  void setBlankThreshold(float p) { builder.setBlankThreshold(p); }
  void setQueueTimeoutMs(uint32_t ms) { writer.setTimeoutMs(ms); }

  uint32_t droppedFrames() const { return writer.droppedFrames(); }
  uint32_t emittedMessages() const { return builder.emittedMessages(); }
  AcousticModel& acousticModel() { return model; }
  FeatureExtractor& featureExtractor() { return features; }

 protected:
  static constexpr int kBlockSamples = FeatureExtractor::kHop;
  static constexpr size_t kBlockBytes = kBlockSamples * sizeof(int16_t);

  FeatureExtractor features;
  AcousticModel model;
  PhonemeQueueWriter writer;
  FrameBuilder builder;
  VAD vad;
  bool vad_enabled = true;
  bool in_utterance = false;
  int16_t block[kBlockSamples];
  size_t block_fill = 0;
  int preroll_frames = 30;
  std::vector<float> preroll_buf;
  int preroll_count = 0, preroll_pos = 0;

  void processBlock() {
    VAD::Event ev = vad_enabled ? vad.update(block, kBlockSamples, FeatureExtractor::kSampleRate) : VAD::None;
    if (ev == VAD::SpeechStart) startUtterance();
    features.write((const uint8_t*)block, kBlockBytes);  // -> onFeatures
    if (ev == VAD::SpeechEnd) endOfUtterance();
  }

  void startUtterance() {
    model.reset();
    builder.reset();
    in_utterance = true;
    // replay the buffered frames (oldest first)
    int start = (preroll_pos - preroll_count + preroll_frames) % (preroll_frames > 0 ? preroll_frames : 1);
    for (int j = 0; j < preroll_count; j++) {
      int idx = (start + j) % preroll_frames;
      runModel(&preroll_buf[(size_t)idx * FeatureExtractor::kMelBins]);
    }
    preroll_count = preroll_pos = 0;
  }

  static void onFeatures(const float* f, int n, void* ref) {
    SpeechToPhonemes* self = (SpeechToPhonemes*)ref;
    if (self->in_utterance) {
      self->runModel(f);
    } else if (self->preroll_frames > 0) {
      memcpy(&self->preroll_buf[(size_t)self->preroll_pos * n], f, n * sizeof(float));
      self->preroll_pos = (self->preroll_pos + 1) % self->preroll_frames;
      if (self->preroll_count < self->preroll_frames) self->preroll_count++;
    }
  }

  void runModel(const float* f) { emitFrames(model.write(f)); }

  void emitFrames(int n) {
    for (int i = 0; i < n; i++) builder.write(model.logProbs(i), model.classCount());
  }
};

}  // namespace tinystt
