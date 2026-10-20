// Step 3: FeatureExtractor on the arduino-audio-tools FFT API, with the FFT
// implementation injected. Every check runs with AudioRealFFT and with the
// test-only AudioNaiveDFT to prove that the selection works.
#include <fstream>

#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "NaiveDFT.h"
#include "TestUtil.h"
#include "TinySTT/FeatureExtractor.h"
#include "TinySTT/VAD.h"

using namespace tinystt;
using FE = FeatureExtractor;

struct Collector {
  std::vector<std::vector<float>> frames;
  static void cb(const float* f, int n, void* ref) {
    ((Collector*)ref)->frames.emplace_back(f, f + n);
  }
};

// same generator as research/features.py test_signal()
static std::vector<int16_t> testSignal(int n = 16000, uint32_t seed = 1234) {
  std::vector<int16_t> out(n);
  uint32_t state = seed;
  for (int i = 0; i < n; i++) {
    state = 1664525u * state + 1013904223u;
    int noise = (int)((state >> 16) & 0x7FFF) - 16384;
    double t = (double)i / FE::kSampleRate;
    double chirp = std::sin(2 * M_PI * (200.0 * t + 1900.0 * t * t));
    double v = std::round(12000.0 * chirp + 0.1 * noise);
    out[i] = (int16_t)std::max(-32768.0, std::min(32767.0, v));
  }
  return out;
}

static std::vector<int16_t> sine(float freq, int n, float amp = 16000) {
  std::vector<int16_t> out(n);
  for (int i = 0; i < n; i++) out[i] = (int16_t)(amp * std::sin(2 * M_PI * freq * i / FE::kSampleRate));
  return out;
}

static Collector run(audio_tools::AudioFFTBase& fft, const std::vector<int16_t>& pcm, int chunk = 0) {
  Collector c;
  FE fe(fft);
  fe.setFrameCallback(Collector::cb, &c);
  fe.begin();
  const uint8_t* p = (const uint8_t*)pcm.data();
  size_t bytes = pcm.size() * 2;
  if (chunk <= 0) chunk = (int)bytes;
  for (size_t pos = 0; pos < bytes; pos += chunk) {
    size_t n = std::min((size_t)chunk, bytes - pos);
    fe.write(p + pos, n);
  }
  fe.end();
  return c;
}

static int expectedFrames(size_t samples) {
  return samples < (size_t)FE::kFFTLength ? 0 : 1 + (int)(samples - FE::kFFTLength) / FE::kHop;
}

// band whose center frequency is closest to `hz`
static int melBandFor(float hz) {
  auto toMel = [](float f) { return 2595.0f * log10f(1.0f + f / 700.0f); };
  float step = (toMel(FE::kMaxFreq) - toMel(FE::kMinFreq)) / (FE::kMelBins + 1);
  int best = 0;
  float best_d = 1e9f;
  for (int i = 0; i < FE::kMelBins; i++) {
    float c = 700.0f * (powf(10.0f, (toMel(FE::kMinFreq) + (i + 1) * step) / 2595.0f) - 1.0f);
    if (fabsf(c - hz) < best_d) {
      best_d = fabsf(c - hz);
      best = i;
    }
  }
  return best;
}

template <class TFFT>
static void checkSinePeak() {
  for (float hz : {300.0f, 1000.0f, 3000.0f, 6000.0f}) {
    TFFT fft;
    auto c = run(fft, sine(hz, 4000));
    CHECK(!c.frames.empty());
    auto& f = c.frames.back();
    int peak = (int)(std::max_element(f.begin(), f.end()) - f.begin());
    CHECK(std::abs(peak - melBandFor(hz)) <= 1);
  }
}

template <class TFFT>
static void checkFrameCount() {
  for (int n : {100, 511, 512, 513, 671, 672, 16000, 16123}) {
    TFFT fft;
    auto c = run(fft, testSignal(n));
    CHECK_EQ((int)c.frames.size(), expectedFrames(n));
  }
}

template <class TFFT>
static void checkChunkInvariance() {
  auto pcm = testSignal(8000);
  TFFT ref_fft;
  auto ref = run(ref_fft, pcm);
  for (int chunk : {2, 6, 320, 1000, 4096}) {
    TFFT fft;
    auto c = run(fft, pcm, chunk);
    CHECK_EQ(c.frames.size(), ref.frames.size());
    bool same = c.frames == ref.frames;
    CHECK(same);
  }
}

TEST(features_sine_peak_realfft) { checkSinePeak<audio_tools::AudioRealFFT>(); }
TEST(features_sine_peak_naive_dft) { checkSinePeak<AudioNaiveDFT>(); }
TEST(features_frame_count_realfft) { checkFrameCount<audio_tools::AudioRealFFT>(); }
TEST(features_frame_count_naive_dft) { checkFrameCount<AudioNaiveDFT>(); }
TEST(features_chunk_invariance_realfft) { checkChunkInvariance<audio_tools::AudioRealFFT>(); }
TEST(features_chunk_invariance_naive_dft) { checkChunkInvariance<AudioNaiveDFT>(); }

TEST(features_no_empty_bands) {
  audio_tools::AudioRealFFT fft;
  auto c = run(fft, testSignal(4000));
  for (auto& f : c.frames)
    for (float v : f) CHECK(v > logf(FE::kLogEps) + 1.0f);
}

TEST(features_ffts_agree) {
  auto pcm = testSignal(4000);
  audio_tools::AudioRealFFT real;
  AudioNaiveDFT naive;
  auto a = run(real, pcm);
  auto b = run(naive, pcm);
  CHECK_EQ(a.frames.size(), b.frames.size());
  double max_diff = 0;
  for (size_t i = 0; i < a.frames.size(); i++)
    for (int j = 0; j < FE::kMelBins; j++)
      max_diff = std::max(max_diff, (double)std::fabs(a.frames[i][j] - b.frames[i][j]));
  CHECK(max_diff < 1e-3);
}

TEST(features_match_python_golden) {
  std::ifstream in(TINYSTT_TEST_DATA "/features_golden.bin", std::ios::binary);
  CHECK(in.good());
  if (!in) return;
  uint32_t hdr[3];
  in.read((char*)hdr, sizeof(hdr));
  std::vector<int16_t> pcm(hdr[0]);
  in.read((char*)pcm.data(), pcm.size() * 2);
  std::vector<float> golden((size_t)hdr[1] * hdr[2]);
  in.read((char*)golden.data(), golden.size() * 4);
  CHECK(in.good());
  CHECK(pcm == testSignal(hdr[0]));  // generators agree
  audio_tools::AudioRealFFT fft;
  auto c = run(fft, pcm);
  CHECK_EQ(c.frames.size(), (size_t)hdr[1]);
  CHECK_EQ((uint32_t)FE::kMelBins, hdr[2]);
  double max_diff = 0, sum_diff = 0;
  size_t max_i = 0, max_j = 0;
  for (size_t i = 0; i < c.frames.size() && i < hdr[1]; i++)
    for (uint32_t j = 0; j < hdr[2]; j++) {
      double d = std::fabs(c.frames[i][j] - golden[i * hdr[2] + j]);
      sum_diff += d;
      if (d > max_diff) {
        max_diff = d;
        max_i = i;
        max_j = j;
      }
    }
  std::printf("  max |c++ - python| = %g at frame %zu band %zu (value %g)\n", max_diff, max_i,
              max_j, golden[max_i * hdr[2] + max_j]);
  // float32 FFT on the device vs float64 numpy: the largest deviations are
  // in very low energy bands (ln x ~ -7), far below training noise
  double mean_diff = sum_diff / ((double)hdr[1] * hdr[2]);
  std::printf("  mean |c++ - python| = %g\n", mean_diff);
  CHECK(mean_diff < 1e-4);
  CHECK(max_diff < 5e-3);
}

TEST(features_normalization) {
  float mean[FE::kMelBins], inv_std[FE::kMelBins];
  for (int j = 0; j < FE::kMelBins; j++) {
    mean[j] = 0.5f * j;
    inv_std[j] = 2.0f;
  }
  auto pcm = testSignal(2000);
  audio_tools::AudioRealFFT f1, f2;
  auto raw = run(f1, pcm);
  Collector c;
  FE fe(f2);
  fe.setNormalization(mean, inv_std);
  fe.setFrameCallback(Collector::cb, &c);
  fe.begin();
  fe.write((const uint8_t*)pcm.data(), pcm.size() * 2);
  CHECK_EQ(c.frames.size(), raw.frames.size());
  for (size_t i = 0; i < c.frames.size(); i++)
    for (int j = 0; j < FE::kMelBins; j++)
      CHECK_NEAR(c.frames[i][j], (raw.frames[i][j] - mean[j]) * inv_std[j], 1e-4);
}

TEST(features_reset_restarts_framing) {
  audio_tools::AudioRealFFT fft;
  Collector c;
  FE fe(fft);
  fe.setFrameCallback(Collector::cb, &c);
  fe.begin();
  auto pcm = testSignal(1000);
  fe.write((const uint8_t*)pcm.data(), pcm.size() * 2);
  CHECK_EQ(fe.frameCount(), (uint32_t)expectedFrames(1000));
  fe.reset();
  CHECK_EQ(fe.frameCount(), 0u);
  fe.write((const uint8_t*)pcm.data(), 400 * 2);  // < one window
  CHECK_EQ(fe.frameCount(), 0u);
}

/// feeds 10 ms blocks of a tone (or silence); returns the time (ms) of SpeechEnd, 0 if none
static uint32_t vadEndMs(VAD& vad, const std::vector<std::pair<uint32_t, bool>>& segments) {
  int16_t block[160];
  uint32_t t = 0;
  for (auto& seg : segments) {
    for (uint32_t ms = 0; ms < seg.first; ms += 10, t += 10) {
      for (int j = 0; j < 160; j++) block[j] = seg.second ? (int16_t)(8000 * sinf(2 * PI * 300 * j / 16000.0f)) : 0;
      if (vad.update(block, 160) == VAD::SpeechEnd) return t + 10;
    }
  }
  return 0;
}

TEST(vad_long_utterance_ends_at_pause) {
  VAD vad;
  vad.begin();
  // 16 s of speech, a 100 ms pause, more speech: ends in the pause, not at 15 s
  uint32_t end = vadEndMs(vad, {{16000, true}, {100, false}, {5000, true}});
  std::printf("  speech end at %u ms\n", (unsigned)end);
  CHECK(end > 16000 && end <= 16100);
}

TEST(vad_hard_limit) {
  VAD vad;
  vad.begin();
  uint32_t end = vadEndMs(vad, {{40000, true}});  // no pause at all
  CHECK(end > 0 && end <= vad.config().hard_max_utterance_ms + 50);
}

TEST_MAIN()
