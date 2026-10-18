// Step 9: end-to-end audio -> text with the trained toy model
// (research/make_toy_task.py): single task, two tasks over the shared SPSC
// queue (TinySTT facade), and split through queue -> bridge -> byte stream ->
// bridge -> queue.
// All three must give identical text.
#include <atomic>
#include <algorithm>
#include <fstream>
#include <sstream>

#include "AudioTools.h"
#include "AudioTools/AudioCodecs/CodecWAV.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "LoopbackStream.h"
#include "TestData.h"
#include "TestUtil.h"
#include "TinySTT.h"

static std::vector<uint8_t> g_model = readTestFile("toy_model.tstt");

/// Collects decoded PCM bytes
struct VectorPrint : public Print {
  std::vector<uint8_t> data;
  size_t write(uint8_t b) override {
    data.push_back(b);
    return 1;
  }
  size_t write(const uint8_t* b, size_t n) override {
    data.insert(data.end(), b, b + n);
    return n;
  }
};

/// WAV file -> 16 kHz mono int16 PCM via audio-tools' WAVDecoder
static std::vector<uint8_t> loadPcm(const std::string& name) {
  auto wav = readTestFile(name);
  VectorPrint pcm;
  audio_tools::WAVDecoder decoder;
  audio_tools::EncodedAudioStream dec(&pcm, &decoder);
  dec.begin();
  for (size_t pos = 0; pos < wav.size(); pos += 1024)
    dec.write(wav.data() + pos, std::min<size_t>(1024, wav.size() - pos));
  dec.end();
  auto info = decoder.audioInfo();
  std::printf("  %s: %d Hz, %d ch, %d bit, %zu bytes\n", name.c_str(), (int)info.sample_rate, info.channels,
              info.bits_per_sample, pcm.data.size());
  if (info.sample_rate != 16000 || info.channels != 1 || info.bits_per_sample != 16) return {};
  return pcm.data;
}

static std::vector<std::vector<std::string>> expectedTexts() {
  std::vector<std::vector<std::string>> out;
  std::ifstream in(TINYSTT_TEST_DATA "/toy_expected.txt");
  std::string line;
  while (std::getline(in, line)) {
    std::vector<std::string> utts;
    std::stringstream ss(line);
    std::string u;
    while (std::getline(ss, u, '|')) utts.push_back(u);
    out.push_back(utts);
  }
  return out;
}

struct Results {
  std::vector<std::string> finals;
  std::vector<float> confidence;
  int partials = 0;
  std::atomic<int> count{0};
  static void cb(const SttResult& r, void* ref) {
    Results* self = (Results*)ref;
    if (r.is_final) {
      self->finals.push_back(r.text());
      self->confidence.push_back(r.confidence());
      self->count++;
    } else {
      self->partials++;
    }
  }
};

/// Single task: write chunks, decode after each chunk
static std::vector<std::string> runSingle(const std::vector<uint8_t>& pcm, float blank_threshold = 0.95f,
                                          Results* out = nullptr) {
  ModelContainer mc;
  mc.begin(g_model.data(), g_model.size());
  audio_tools::AudioRealFFT fft;
  STTPipeline stt(fft);
  if (!stt.begin(mc)) return {"<begin failed>"};
  stt.speechToPhonemes().setBlankThreshold(blank_threshold);
  Results local;
  Results& res = out ? *out : local;
  stt.phonemesToText().setResultCallback(Results::cb, &res);
  for (size_t pos = 0; pos < pcm.size(); pos += 1000) {
    stt.write(pcm.data() + pos, std::min<size_t>(1000, pcm.size() - pos));
    stt.copy();
  }
  stt.endOfUtterance();  // no-op if the VAD already ended it
  stt.copy();
  return res.finals;
}

static std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (auto& x : v) s += (s.empty() ? "" : "|") + x;
  return s;
}

/// Non-empty results as printed by TinySTT: one per line
static std::string lines(const std::vector<std::string>& v) {
  std::string s;
  for (auto& x : v)
    if (!x.empty()) s += x + "\n";
  return s;
}

static int lineCount(const std::string& text) { return (int)std::count(text.begin(), text.end(), '\n'); }

static std::vector<std::string> words(const std::vector<std::string>& utterances) {
  std::vector<std::string> out;
  for (auto& u : utterances) {
    std::istringstream in(u);
    std::string w;
    while (in >> w) out.push_back(w);
  }
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

/// Reference result of every clip: one loop (STTPipeline), computed once.
/// The other setups (tasks, facade, split) must reproduce it exactly.
static const std::vector<std::vector<std::string>>& reference() {
  static std::vector<std::vector<std::string>> ref;
  if (ref.empty())
    for (size_t i = 0; i < 3; i++) ref.push_back(runSingle(loadPcm("toy_utt" + std::to_string(i) + ".wav")));
  return ref;
}

TEST(pipeline_single_task_accuracy) {
  // The toy model is a small, imperfect model: accuracy is checked as a word
  // error rate; exact equality is required between the setups (below).
  CHECK(!g_model.empty());
  auto expected = expectedTexts();
  CHECK_EQ(expected.size(), 3u);
  int errors = 0, total = 0;
  for (size_t i = 0; i < expected.size(); i++) {
    Results res;
    auto got = runSingle(loadPcm("toy_utt" + std::to_string(i) + ".wav"), 0.95f, &res);
    int e = editDistance(words(expected[i]), words(got));
    std::printf("  clip %zu: '%s' (%d word errors)\n", i, join(got).c_str(), e);
    errors += e;
    total += (int)words(expected[i]).size();
    CHECK(res.partials > 0);
    CHECK(got == reference()[i]);  // deterministic
  }
  double wer = (double)errors / total;
  std::printf("  WER %.3f (%d / %d words)\n", wer, errors, total);
  CHECK(wer <= 0.25);
}

TEST(pipeline_vad_segments_utterances) {
  // clip 2 holds two utterances separated by 1 s of silence
  auto got = runSingle(loadPcm("toy_utt2.wav"));
  CHECK_EQ(got.size(), 2u);
}

TEST(pipeline_blank_skipping_same_text) {
  auto pcm = loadPcm("toy_utt0.wav");
  CHECK_STR(join(runSingle(pcm, 0.95f)), join(runSingle(pcm, 1.0f)));
}

/// Print that collects the text lines (TinySTT output)
struct LinePrint : public Print {
  std::string text;
  std::atomic<int> lines{0};
  size_t write(uint8_t c) override {
    if (c == '\r') return 1;
    text += (char)c;
    if (c == '\n') lines++;
    return 1;
  }
};

/// Runs a clip through TinySTT with a StreamCopy (as in a sketch)
static std::string runTinySTT(const std::vector<uint8_t>& pcm, int expected_lines, const char* grammar = nullptr,
                              uint32_t* dropped = nullptr) {
  ModelContainer mc;
  mc.begin(g_model.data(), g_model.size());
  audio_tools::AudioRealFFT fft;
  audio_tools::MemoryStream audio(pcm.data(), (int)pcm.size());
  audio.begin();
  LinePrint out;
  TinySTT stt(fft, mc, out);
  if (grammar && !stt.phonemesToText().setGrammar(grammar)) return "<grammar error>";
  if (!stt.begin()) return "<begin failed>";
  audio_tools::StreamCopy copier(stt, audio);
  while (copier.copy() > 0) {
  }
  uint32_t start = millis();
  while (out.lines.load() < expected_lines && millis() - start < 20000) delay(5);
  if (dropped) *dropped = stt.speechToPhonemes().droppedFrames();
  stt.end();
  return out.text;
}

TEST(tinystt_facade_stream_copy) {
  // TinySTT is an AudioOutput: StreamCopy from the source, begin() starts
  // the decoder task, final results are printed as lines - exactly the text
  // of the one-loop reference
  for (size_t i = 0; i < 3; i++) {
    std::string want = lines(reference()[i]);
    uint32_t dropped = 1;
    CHECK_STR(runTinySTT(loadPcm("toy_utt" + std::to_string(i) + ".wav"), lineCount(want), nullptr, &dropped), want);
    CHECK_EQ(dropped, 0u);
  }
}

TEST(tinystt_facade_grammar_before_begin) {
  // configured before begin(): must survive begin() and use the dictionary
  CHECK_STR(runTinySTT(loadPcm("toy_utt0.wav"), 1, "turn (on | off) the (light | fan)"), "turn on the light\n");
  ModelContainer mc;
  audio_tools::AudioRealFFT fft;
  LinePrint out;
  TinySTT stt(fft, mc, out);
  CHECK(!stt.phonemesToText().setGrammar("turn (on | off"));  // syntax checked at once
}

TEST(tinystt_facade_restart) {
  // end() stops the decoder task cooperatively; begin() works again
  auto pcm = loadPcm("toy_utt1.wav");
  ModelContainer mc;
  mc.begin(g_model.data(), g_model.size());
  audio_tools::AudioRealFFT fft;
  LinePrint out;
  TinySTT stt(fft, mc, out);
  for (int round = 0; round < 3; round++) {
    CHECK(stt.begin());
    audio_tools::MemoryStream audio(pcm.data(), (int)pcm.size());
    audio.begin();
    audio_tools::StreamCopy copier(stt, audio);
    while (copier.copy() > 0) {
    }
    uint32_t start = millis();
    while (out.lines.load() < (round + 1) * lineCount(lines(reference()[1])) && millis() - start < 20000) delay(5);
    stt.end();
    CHECK(!stt.isActive());
  }
  std::string one = lines(reference()[1]);
  CHECK(!one.empty());
  CHECK_STR(out.text, one + one + one);
}

TEST(tinystt_facade_end_flushes_last_result) {
  // audio that ends without trailing silence: end() must still print it
  auto pcm = loadPcm("toy_utt0.wav");
  pcm.resize(pcm.size() - 16000 * 2 * 9 / 10);  // cut most of the 1 s tail
  ModelContainer mc;
  mc.begin(g_model.data(), g_model.size());
  audio_tools::AudioRealFFT fft;
  LinePrint out;
  TinySTT stt(fft, mc, out);
  CHECK(stt.begin());
  stt.write(pcm.data(), pcm.size());
  stt.end();
  std::string want = lines(runSingle(pcm));  // runSingle also ends the utterance explicitly
  CHECK(!want.empty());
  CHECK_STR(out.text, want);
}

TEST(tinystt_facade_requires_model_and_format) {
  ModelContainer empty;
  audio_tools::AudioRealFFT fft;
  LinePrint out;
  TinySTT stt(fft, empty, out);
  CHECK(!stt.begin());
  CHECK(!stt.isActive());
  uint8_t pcm[64] = {0};
  CHECK_EQ(stt.write(pcm, sizeof(pcm)), 0u);  // not started
  stt.setAudioInfo(audio_tools::AudioInfo(44100, 2, 16));  // rejected
  CHECK_EQ((int)stt.audioInfo().sample_rate, 16000);
}

TEST(pipeline_split_over_stream) {
  // device A: SpeechToPhonemes -> queue A -> QueueToStream -> "wire"
  // device B: "wire" -> StreamToQueue -> queue B -> PhonemesToText
  for (size_t i = 0; i < 3; i++) {
    auto pcm = loadPcm("toy_utt" + std::to_string(i) + ".wav");
    ModelContainer mc;
    mc.begin(g_model.data(), g_model.size());
    audio_tools::AudioRealFFT fft;
    PhonemeQueue queue_a(64), queue_b(64);
    SpeechToPhonemes stp(fft, queue_a);
    CHECK(stp.begin(mc));
    LoopbackStream wire(5);  // fragmented transport
    QueueToStream sender(queue_a, wire);
    StreamToQueue receiver(wire, queue_b);
    PhonemesToText ptt(queue_b);
    CHECK(ptt.begin(mc));
    Results res;
    ptt.setResultCallback(Results::cb, &res);
    for (size_t pos = 0; pos < pcm.size(); pos += 640) {
      stp.write(pcm.data() + pos, std::min<size_t>(640, pcm.size() - pos));
      sender.copy();
      receiver.copy();
      ptt.copy();
    }
    stp.endOfUtterance();
    sender.copy();
    while (wire.available()) receiver.copy();
    ptt.copy();
    CHECK_STR(join(res.finals), join(reference()[i]));  // identical to one loop
    CHECK_EQ(receiver.protocolState().crcErrors(), 0u);
  }
}

TEST(phoneme_output_from_speech_to_phonemes) {
  // step 1 printing straight into a PhonemeOutput: one line per utterance,
  // only phoneme symbols
  auto expected = expectedTexts();
  for (size_t i = 0; i < expected.size(); i++) {
    auto pcm = loadPcm("toy_utt" + std::to_string(i) + ".wav");
    ModelContainer mc;
    mc.begin(g_model.data(), g_model.size());
    audio_tools::AudioRealFFT fft;
    LinePrint out;
    PhonemeOutput phonemes(out);
    SpeechToPhonemes stp(fft, phonemes);
    CHECK(stp.begin(mc));
    stp.write(pcm.data(), pcm.size());
    stp.endOfUtterance();
    std::printf("  clip %zu: %s", i, out.text.c_str());
    CHECK(out.lines.load() >= 1 && out.lines.load() <= (int)expected[i].size());
    std::istringstream in(out.text);
    std::string sym;
    while (in >> sym) CHECK(Phonemes::id(sym.c_str()) >= Phonemes::kFirstPhone);
  }
}

TEST(pipeline_grammar_mode) {
  auto pcm = loadPcm("toy_utt0.wav");
  ModelContainer mc;
  mc.begin(g_model.data(), g_model.size());
  audio_tools::AudioRealFFT fft;
  STTPipeline stt(fft);
  CHECK(stt.begin(mc));
  CHECK(stt.phonemesToText().setGrammar("turn (on | off) the (light | fan)"));
  Results res;
  stt.phonemesToText().setResultCallback(Results::cb, &res);
  stt.write(pcm.data(), pcm.size());
  stt.endOfUtterance();
  stt.copy();
  CHECK_STR(join(res.finals), "turn on the light");
}

TEST(pipeline_silence_produces_nothing) {
  std::vector<uint8_t> silence(16000 * 2 * 2, 0);  // 2 s
  CHECK(runSingle(silence).empty());
}

TEST_MAIN()
