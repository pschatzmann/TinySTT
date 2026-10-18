/**
 * On-device benchmark of every TinySTT stage (results: docs/performance.md).
 *
 * Model container (e.g. the production-shape benchmark container from
 *   python research/make_bench_model.py --out bench.tstt --no-float32):
 * - ESP32: memory-mapped from the 'tinystt' flash partition (partitions.csv):
 *     esptool --port /dev/ttyACM0 write-flash 0x310000 bench.tstt
 * - other boards (e.g. RP2350 with PSRAM): /tinystt.tstt on LittleFS.
 * Runs every 15 s.
 */
#define TINYSTT_PROFILE
#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "TinySTT.h"
#if defined(ESP32)
#include "AudioTools/FFT/AudioEspressifFFT.h"
ModelPartition partition;
#else
#include "LittleFS.h"
#endif

ModelContainer model;
std::vector<int16_t> pcm;
const int kSeconds = 5;

static uint32_t feature_frames = 0;
static void countFrame(const float*, int, void*) { feature_frames++; }

float rtf(uint32_t us) { return us / (kSeconds * 1e6f); }

void benchFeatures(AudioFFTBase& fft, const char* name) {
  FeatureExtractor fe(fft);
  fe.setFrameCallback(countFrame);
  fe.begin();
  feature_frames = 0;
  uint32_t t0 = micros();
  fe.write((const uint8_t*)pcm.data(), pcm.size() * 2);
  uint32_t t = micros() - t0;
  fe.end();
  Serial.printf("features %-26s %8.1f ms  RTF %.3f  (%u frames)\n", name, t / 1000.0f, rtf(t),
                (unsigned)feature_frames);
}

void benchModel(AcousticModel::Precision precision, AcousticModel::Placement placement, const char* name,
                int batch = 1) {
  AcousticModel m;
  m.setBatchFrames(batch);
  if (!m.begin(model, precision, placement)) {
    Serial.printf("model %s: not available\n", name);
    return;
  }
  // 5 s of features = 500 frames (synthetic features: timing only)
  std::vector<float> f(80);
  uint32_t outputs = 0;
  m.resetProfile();
  uint32_t t0 = micros();
  for (int i = 0; i < kSeconds * 100; i++) {
    for (int j = 0; j < 80; j++) f[j] = (float)((i * 31 + j * 17) % 100) / 50.0f - 1.0f;
    outputs += m.write(f.data());
  }
  outputs += m.flush();
  uint32_t t = micros() - t0;
  Serial.printf("model    %-26s %8.1f ms  RTF %.3f  (%.2f ms/frame: conv %.2f, gru %.2f, out %.2f; %uK internal, %uK PSRAM)\n",
                name, t / 1000.0f, rtf(t), t / 1000.0f / outputs, m.prof_conv_us / 1000.0f / outputs,
                m.prof_gru_us / 1000.0f / outputs, m.prof_out_us / 1000.0f / outputs,
                (unsigned)(m.internalWeightBytes() / 1024), (unsigned)(m.psramWeightBytes() / 1024));
}

void benchPipeline() {
  AudioRealFFT fft;
  PhonemeQueue queue(1024);
  SpeechToPhonemes stp(fft, queue);
  stp.setBatchFrames(4);
  stp.begin(model);
  stp.setVADEnabled(false);
  uint32_t t0 = micros();
  stp.write((const uint8_t*)pcm.data(), pcm.size() * 2);
  stp.endOfUtterance();
  uint32_t t1 = micros() - t0;
  PhonemesToText ptt(queue);
  ptt.begin(model);
  t0 = micros();
  size_t messages = ptt.copy();
  uint32_t t2 = micros() - t0;
  Serial.printf("stage 1  %-26s %8.1f ms  RTF %.3f\n", stp.acousticModel().isQuantized() ? "(int8)" : "(f32)",
                t1 / 1000.0f, rtf(t1));
  Serial.printf("stage 2  %-26s %8.1f ms  RTF %.3f  (%u messages, %d words)\n", "decoder", t2 / 1000.0f, rtf(t2),
                (unsigned)messages, ptt.lexicon().staticWordCount());
}

void setup() {
  Serial.begin(115200);
  delay(3000);
#if defined(ESP32)
  if (!partition.begin(model)) {
    Serial.println("no model container in partition 'tinystt'");
    return;
  }
#else
  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");
  if (!file || model.begin(file, file.size()) != ModelContainer::OK) {
    Serial.println("could not load /tinystt.tstt");
    return;
  }
#endif
  Serial.printf("container: %u bytes, %d sections\n", (unsigned)model.size(), model.sectionCount());
  pcm.resize(16000 * kSeconds);
  for (size_t j = 0; j < pcm.size(); j++)  // chirp + noise
    pcm[j] = (int16_t)(8000 * sinf(2 * PI * (200.0f + 0.05f * j) * j / 16000.0f) + random(-500, 500));
}

void loop() {
  if (!model) {
    delay(1000);
    return;
  }
#if defined(ESP32)
  Serial.printf("--- TinySTT benchmark, %d s audio, CPU %u MHz, int8 kernel %s, free heap %u (largest block %u), free PSRAM %u\n",
                kSeconds, (unsigned)getCpuFrequencyMhz(), simdS8Name(), (unsigned)ESP.getFreeHeap(),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)ESP.getFreePsram());
#else
  Serial.printf("--- TinySTT benchmark, %d s audio, int8 kernel %s\n", kSeconds, simdS8Name());
#endif
  AudioRealFFT real_fft;
  benchFeatures(real_fft, "AudioRealFFT");
#if defined(ESP32)
  AudioEspressifFFT esp_fft;
  benchFeatures(esp_fft, "AudioEspressifFFT");
#endif
  using P = AcousticModel::Precision;
  using L = AcousticModel::Placement;
  benchModel(P::Float32, L::InPlace, "float32 flash");
  benchModel(P::Int8, L::InPlace, "int8 weights flash");
  benchModel(P::Int8Activations, L::InPlace, "int8 SIMD flash");
  benchModel(P::Int8Activations, L::ExternalRAM, "int8 SIMD PSRAM");
  benchModel(P::Int8Activations, L::Auto, "int8 SIMD auto");
  benchModel(P::Int8Activations, L::Auto, "int8 SIMD auto batch 2", 2);
  benchModel(P::Int8Activations, L::Auto, "int8 SIMD auto batch 4", 4);
  benchModel(P::Int8Activations, L::Auto, "int8 SIMD auto batch 8", 8);
  benchPipeline();
  Serial.println("--- done");
  delay(15000);
}
