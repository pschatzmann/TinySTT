// tinystt_benchmark: desktop counterpart of examples/timing_benchmark.
// Measures features, the acoustic model (precisions, batch sizes), stage 1
// and the decoder on 5 s of synthetic audio (results: docs/performance.md).
//
//   python research/make_bench_model.py --out bench.tstt
//   ./build/desktop/tinystt_benchmark bench.tstt
#define TINYSTT_PROFILE
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "TinySTT.h"

static const int kSeconds = 5;
static const int kRepeats = 5;  // best of kRepeats after one warm-up run (CPU clock scaling)

/// Runs f() once as warm-up, then kRepeats times; returns the fastest run in ms
template <class F>
static double bestOf(F f) {
  f();
  double best = 1e30;
  for (int r = 0; r < kRepeats; r++) best = std::min(best, f());
  return best;
}
static std::vector<int16_t> pcm;
static ModelContainer model;

static double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static double rtf(double ms) { return ms / (kSeconds * 1000.0); }

static uint32_t feature_frames = 0;
static void countFrame(const float*, int, void*) { feature_frames++; }

static void benchFeatures() {
  double t = bestOf([] {
    audio_tools::AudioRealFFT fft;
    FeatureExtractor fe(fft);
    fe.setFrameCallback(countFrame);
    fe.begin();
    feature_frames = 0;
    double t0 = nowMs();
    fe.write((const uint8_t*)pcm.data(), pcm.size() * 2);
    return nowMs() - t0;
  });
  printf("features %-26s %8.2f ms  RTF %.4f  (%u frames)\n", "AudioRealFFT", t, rtf(t), (unsigned)feature_frames);
}

static void benchModel(AcousticModel::Precision p, const char* name, int batch) {
  AcousticModel m;
  m.setBatchFrames(batch);
  if (!m.begin(model, p)) {
    printf("model %s: not available\n", name);
    return;
  }
  std::vector<float> f(80);
  uint32_t outputs = 0;
  uint32_t conv = 0, gru = 0, out = 0;
  double best = 1e30;
  double t = bestOf([&] {
    m.reset();
    m.resetProfile();
    outputs = 0;
    double t0 = nowMs();
    for (int i = 0; i < kSeconds * 100; i++) {
      for (int j = 0; j < 80; j++) f[j] = (float)((i * 31 + j * 17) % 100) / 50.0f - 1.0f;
      outputs += m.write(f.data());
    }
    outputs += m.flush();
    double elapsed = nowMs() - t0;
    if (elapsed < best) {  // stage profile of the fastest run
      best = elapsed;
      conv = m.prof_conv_us;
      gru = m.prof_gru_us;
      out = m.prof_out_us;
    }
    return elapsed;
  });
  printf("model    %-26s %8.2f ms  RTF %.4f  (%.3f ms/frame: conv %.3f, gru %.3f, out %.3f)\n", name, t, rtf(t),
         t / outputs, conv / 1000.0 / outputs, gru / 1000.0 / outputs, out / 1000.0 / outputs);
}

static void benchPipeline() {
  size_t messages = 0;
  int words = 0;
  double t2 = 1e30;
  double t1 = bestOf([&] {
    audio_tools::AudioRealFFT fft;
    PhonemeQueue queue(1024);
    SpeechToPhonemes stp(fft, queue);
    stp.setBatchFrames(4);
    stp.begin(model);
    stp.setVADEnabled(false);
    double t0 = nowMs();
    stp.write((const uint8_t*)pcm.data(), pcm.size() * 2);
    stp.endOfUtterance();
    double stage1 = nowMs() - t0;
    PhonemesToText ptt(queue);
    ptt.begin(model);
    t0 = nowMs();
    messages = ptt.copy();
    t2 = std::min(t2, nowMs() - t0);
    words = ptt.lexicon().staticWordCount();
    return stage1;
  });
  printf("stage 1  %-26s %8.2f ms  RTF %.4f\n", "(int8 activations, batch 4)", t1, rtf(t1));
  printf("stage 2  %-26s %8.2f ms  RTF %.4f  (%u messages, %d words)\n", "decoder", t2, rtf(t2), (unsigned)messages,
         words);
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "bench.tstt";
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "usage: tinystt_benchmark <container.tstt> (see research/make_bench_model.py)\n");
    return 2;
  }
  std::vector<uint8_t> data;
  uint8_t buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
  fclose(f);
  if (model.begin(data.data(), data.size()) != ModelContainer::OK) {
    fprintf(stderr, "invalid container %s\n", path);
    return 1;
  }
  audio_tools::AudioToolsLogger.begin(Serial, audio_tools::AudioLogger::Error);
  pcm.resize(16000 * kSeconds);
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> noise(-500, 499);
  for (size_t j = 0; j < pcm.size(); j++)  // chirp + noise, as on the device
    pcm[j] = (int16_t)(8000 * sinf(2 * (float)M_PI * (200.0f + 0.05f * j) * j / 16000.0f) + noise(rng));

  printf("--- TinySTT desktop benchmark, %d s audio, best of %d runs, container %u bytes, int8 kernel: %s\n",
         kSeconds, kRepeats, (unsigned)data.size(), simdS8Name());
  benchFeatures();
  using P = AcousticModel::Precision;
  benchModel(P::Float32, "float32", 1);
  benchModel(P::Int8, "int8 weights", 1);
  benchModel(P::Int8Activations, "int8 activations", 1);
  benchModel(P::Int8Activations, "int8 activations batch 4", 4);
  benchModel(P::Int8Activations, "int8 activations batch 8", 8);
  benchPipeline();
  printf("--- done\n");
  return 0;
}
