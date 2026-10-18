#pragma once
#include <math.h>

#include "AudioTools.h"
#include "AudioTools/FFT/AudioFFT.h"

namespace tinystt {

/**
 * @brief Streaming log-mel features on top of the arduino-audio-tools FFT API.
 *
 * The FFT implementation is selected by the caller, who passes any
 * audio_tools::AudioFFTBase subclass (AudioRealFFT, AudioKissFFT,
 * AudioEspressifFFT, AudioESP32FFT, AudioCmsisFFT ...). The extractor
 * configures that object (16 kHz mono int16, 512 point frames, 160 sample /
 * 10 ms hop, audio-tools' Hann window cached by BufferedWindow) and owns its callback, so the FFT
 * object must be dedicated to this extractor.
 *
 * For every hop it computes fft.toMEL(80, 0, 8000, power) -> ln(x + eps) ->
 * optional (x - mean) * inv_std and hands the 80 values to the frame callback.
 * research/features.py reproduces this exactly for training.
 */
class FeatureExtractor {
 public:
  static constexpr int kSampleRate = 16000;
  static constexpr int kFFTLength = 512;
  static constexpr int kHop = 160;
  static constexpr int kMelBins = 80;
  static constexpr float kMinFreq = 0.0f;
  static constexpr float kMaxFreq = 8000.0f;
  static constexpr float kLogEps = 1e-6f;

  using FrameCallback = void (*)(const float* features, int count, void* ref);

  explicit FeatureExtractor(audio_tools::AudioFFTBase& fft) : p_fft(&fft), window(&hann) {}

  /// Called once per 10 ms hop with kMelBins values
  void setFrameCallback(FrameCallback cb, void* ref = nullptr) {
    frame_callback = cb;
    callback_ref = ref;
  }

  /// Per-bin normalization (from the model container); nullptr disables it.
  /// Both arrays must hold kMelBins values and stay valid.
  void setNormalization(const float* mean, const float* inv_std) {
    p_mean = mean;
    p_inv_std = inv_std;
  }

  /// true (default): mel of the power spectrum |X|^2, false: of |X|
  void setPower(bool flag) { power = flag; }

  bool begin() {
    auto cfg = p_fft->defaultConfig();
    cfg.sample_rate = kSampleRate;
    cfg.channels = 1;
    cfg.bits_per_sample = 16;
    cfg.length = kFFTLength;
    cfg.stride = kHop;
    cfg.window_function = &window;
    cfg.callback = onFFT;
    cfg.ref = this;
    frames = 0;
    if (!p_fft->begin(cfg)) return false;
    // build the cached mel filterbank now, not during the first utterance
    p_fft->toMEL(kMelBins, kMinFreq, kMaxFreq, power);
    return true;
  }

  void end() { p_fft->end(); }

  /// Restarts framing (e.g. between utterances): the next frame needs a full
  /// kFFTLength samples again
  void reset() {
    p_fft->reset();
    frames = 0;
  }

  /// 16 kHz mono int16 PCM
  size_t write(const uint8_t* data, size_t len) { return p_fft->write(data, len); }

  /// Number of frames produced since begin()/reset()
  uint32_t frameCount() const { return frames; }

  audio_tools::AudioFFTBase& fft() { return *p_fft; }

 protected:
  audio_tools::AudioFFTBase* p_fft;
  audio_tools::Hann hann;
  audio_tools::BufferedWindow window;  ///< caches the Hann factors (no cos() per sample)
  FrameCallback frame_callback = nullptr;
  void* callback_ref = nullptr;
  const float* p_mean = nullptr;
  const float* p_inv_std = nullptr;
  bool power = true;
  uint32_t frames = 0;
  float features[kMelBins];

  static void onFFT(audio_tools::AudioFFTBase& fft) {
    FeatureExtractor* self = (FeatureExtractor*)fft.config().ref;
    self->process();
  }

  void process() {
    const float* mel = p_fft->toMEL(kMelBins, kMinFreq, kMaxFreq, power);
    if (mel == nullptr) return;
    for (int j = 0; j < kMelBins; j++) {
      float v = logf(mel[j] + kLogEps);
      if (p_mean != nullptr) v = (v - p_mean[j]) * p_inv_std[j];
      features[j] = v;
    }
    frames++;
    if (frame_callback != nullptr) frame_callback(features, kMelBins, callback_ref);
  }
};

}  // namespace tinystt
