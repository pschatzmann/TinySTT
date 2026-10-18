// Optional TTS -> STT round trip (CMake: -DTINYSTT_ROUNDTRIP=ON, needs the
// sibling TinyTTS library): synthesizes phrases with TinyTTS, transcribes
// them with TinySTT and checks the word error rate.
//
// Needs a model trained on real speech: set TINYSTT_MODEL to the container
// (research/kaggle_train.ipynb). Without it the test reports SKIP and passes.
#include <cstdlib>
#include <sstream>

#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "TestUtil.h"
#include "TinySTT.h"
// TinyTTS
#include "TinyTTS.h"
#include "TinyTTS/data/default_cmudict_slim_data.h"
#include "TinyTTS/data/default_dictionary_model_data.h"
#include "TinyTTS/data/default_weights_data.h"

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream in(s);
  std::string w;
  while (in >> w) out.push_back(w);
  return out;
}

static int editDistance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
  for (size_t j = 0; j <= b.size(); j++) prev[j] = (int)j;
  for (size_t i = 1; i <= a.size(); i++) {
    cur[0] = (int)i;
    for (size_t j = 1; j <= b.size(); j++)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1])});
    prev = cur;
  }
  return prev[b.size()];
}

/// float PCM at `rate` -> int16 at 16 kHz (linear interpolation)
static std::vector<int16_t> to16k(const std::vector<float>& x, int rate) {
  size_t n = (size_t)((double)x.size() * 16000 / rate);
  std::vector<int16_t> out(n);
  for (size_t i = 0; i < n; i++) {
    double t = (double)i * rate / 16000;
    size_t k = (size_t)t;
    double f = t - k;
    double v = k + 1 < x.size() ? x[k] * (1 - f) + x[k + 1] * f : x.back();
    out[i] = (int16_t)std::max(-32768.0, std::min(32767.0, v * 32767.0));
  }
  return out;
}

TEST(roundtrip_tts_to_stt) {
  const char* model_path = std::getenv("TINYSTT_MODEL");
  if (model_path == nullptr) {
    std::printf("  SKIP: set TINYSTT_MODEL to a trained container\n");
    return;
  }
  FILE* f = fopen(model_path, "rb");
  CHECK(f != nullptr);
  if (!f) return;
  std::vector<uint8_t> img;
  uint8_t buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) img.insert(img.end(), buf, buf + n);
  fclose(f);
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), img.size()), ModelContainer::OK);

  tinytts::TinyTTS tts;
  tts.setWeights(tinytts::default_weights, tinytts::default_weights_len);
  tts.setDictionary(tinytts::default_cmudict_slim, tinytts::default_cmudict_slim_len);
  tts.setDictionaryModel(tinytts::default_dictionary_model, tinytts::default_dictionary_model_len);
  std::vector<float> audio;
  CHECK(tts.begin([&](const float* s, size_t c) { audio.insert(audio.end(), s, s + c); }));

  const char* phrases[] = {"turn on the light", "what time is it", "hello world", "the weather is nice today"};
  int errors = 0, words = 0;
  for (const char* p : phrases) {
    audio.clear();
    tts.speak(p);
    auto pcm = to16k(audio, tts.getAudioSampleRate());
    pcm.insert(pcm.begin(), 8000, 0);  // 0.5 s lead silence
    audio_tools::AudioRealFFT fft;
    STTPipeline stt(fft);
    CHECK(stt.begin(mc));
    stt.write((const uint8_t*)pcm.data(), pcm.size() * 2);
    stt.endOfUtterance();
    stt.copy();
    std::string got = stt.phonemesToText().lastResult().text();
    int e = editDistance(split(p), split(got));
    std::printf("  '%s' -> '%s' (%d errors)\n", p, got.c_str(), e);
    errors += e;
    words += (int)split(p).size();
  }
  double wer = (double)errors / words;
  std::printf("  WER %.2f\n", wer);
  CHECK(wer <= 0.35);
}

TEST_MAIN()
