// tinystt: desktop command line front end for TinySTT (see docs/platform-desktop.md).
//
//   tinystt --model tinystt.tstt speech.wav              -> prints the text
//   tinystt --model m.tstt --phonemes-only speech.wav | tinystt --model m.tstt --from-phonemes
//   tinystt --model m.tstt --grammar "turn (on|off) the light" speech.wav
//   tinystt --model m.tstt --mic                          (built with -DTINYSTT_DESKTOP_MIC=ON)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "AudioTools.h"
#include "AudioTools/AudioCodecs/CodecWAV.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "TinySTT.h"
#ifdef TINYSTT_HAVE_MIC
#include "AudioTools/AudioLibs/MiniAudioStream.h"
#endif

/// Print to a C stream (stdout for results/phonemes, stderr for logs)
struct FilePrint : public Print {
  explicit FilePrint(FILE* f) : file(f) {}
  size_t write(uint8_t b) override { return fputc(b, file) == EOF ? 0 : 1; }
  size_t write(const uint8_t* b, size_t n) override {
    size_t r = fwrite(b, 1, n, file);
    fflush(file);
    return r;
  }
  FILE* file;
};

/// Collects bytes (decoded WAV PCM)
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

struct Options {
  std::string model, input, grammar, words;
  std::vector<std::pair<std::string, float>> hotwords;
  bool phonemes_only = false, from_phonemes = false, text_phonemes = false, mic = false;
  bool no_vad = false, verbose = false;
  int nbest = 1, beam = 16;
  float lm_weight = -1.0f, word_bonus = -1000.0f;
  int word_end_beam = -1;  // < defaults: keep the decoder defaults
  float blank_threshold = 0.95f, min_confidence = 0.0f;
};

static void usage() {
  fprintf(stderr,
          "usage: tinystt --model FILE [options] [input.wav | -]\n"
          "  input: a WAV file (any rate/channels) or '-' for raw 16 kHz mono s16le on stdin\n"
          "  --phonemes-only       run SpeechToPhonemes only, binary phoneme frames to stdout\n"
          "  --text-phonemes       like --phonemes-only but human readable\n"
          "  --from-phonemes       run PhonemesToText only, binary phoneme frames from stdin\n"
          "  --grammar TEXT        command grammar, e.g. \"turn (on|off) the (light|fan)\"\n"
          "  --words a,b,c         only recognize these words (word-list mode)\n"
          "  --hotword WORD[:B]    boost a word (log bonus B, default 3)\n"
          "  --nbest N             print N hypotheses with confidences\n"
          "  --beam N              beam size (default 16)\n"
          "  --word-end-beam N     word boundary hypotheses always kept (default 8)\n"
          "  --lm-weight W         language model weight\n"
          "  --word-bonus B        word insertion bonus\n"
          "  --blank-threshold P   blank skipping threshold (default 0.95, 1 = off)\n"
          "  --min-confidence C    reject final results below C\n"
          "  --no-vad              no endpointing: the whole input is one utterance\n"
          "  --mic                 live microphone (if built with TINYSTT_DESKTOP_MIC)\n"
          "  --verbose             log details to stderr\n");
}

static bool parse(int argc, char** argv, Options& o) {
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", name);
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") o.model = next("--model");
    else if (a == "--phonemes-only") o.phonemes_only = true;
    else if (a == "--text-phonemes") o.phonemes_only = o.text_phonemes = true;
    else if (a == "--from-phonemes") o.from_phonemes = true;
    else if (a == "--grammar") o.grammar = next("--grammar");
    else if (a == "--words") o.words = next("--words");
    else if (a == "--hotword") {
      std::string h = next("--hotword");
      size_t c = h.find(':');
      o.hotwords.push_back({h.substr(0, c), c == std::string::npos ? 3.0f : (float)atof(h.c_str() + c + 1)});
    } else if (a == "--nbest") o.nbest = atoi(next("--nbest"));
    else if (a == "--beam") o.beam = atoi(next("--beam"));
    else if (a == "--lm-weight") o.lm_weight = (float)atof(next("--lm-weight"));
    else if (a == "--word-end-beam") o.word_end_beam = atoi(next("--word-end-beam"));
    else if (a == "--word-bonus") o.word_bonus = (float)atof(next("--word-bonus"));
    else if (a == "--blank-threshold") o.blank_threshold = (float)atof(next("--blank-threshold"));
    else if (a == "--min-confidence") o.min_confidence = (float)atof(next("--min-confidence"));
    else if (a == "--no-vad") o.no_vad = true;
    else if (a == "--mic") o.mic = true;
    else if (a == "--verbose") o.verbose = true;
    else if (a == "-h" || a == "--help") return false;
    else if (!a.empty() && a[0] == '-' && a != "-") {
      fprintf(stderr, "unknown option %s\n", a.c_str());
      return false;
    } else o.input = a;
  }
  if (o.model.empty() && !o.from_phonemes && o.grammar.empty() && o.words.empty()) {
    fprintf(stderr, "--model is required\n");
    return false;
  }
  if (o.input.empty() && !o.mic && !o.from_phonemes) o.input = "-";
  return true;
}

static std::vector<uint8_t> readFile(const std::string& path) {
  std::vector<uint8_t> out;
  FILE* f = path == "-" ? stdin : fopen(path.c_str(), "rb");
  if (!f) return out;
  uint8_t buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  if (f != stdin) fclose(f);
  return out;
}

static Options g_opt;

static void printResult(const SttResult& r, void*) {
  if (!r.is_final) {
    if (g_opt.verbose && r.nbest.size()) fprintf(stderr, "... %s\n", r.text());
    return;
  }
  if (r.nbest.empty()) return;
  if (!r.accepted) {
    fprintf(stderr, "(rejected, confidence %.2f: %s)\n", r.confidence(), r.text());
    return;
  }
  if (g_opt.nbest <= 1) {
    printf("%s\n", r.text());
  } else {
    for (auto& h : r.nbest) printf("%.3f\t%s\n", h.confidence, h.text.c_str());
    printf("\n");
  }
  fflush(stdout);
}

/// Writes any WAV/PCM input into `out` as 16 kHz mono int16 (chunked)
static bool feedAudio(const Options& o, Print& out) {
  auto bytes = readFile(o.input);
  if (bytes.empty()) {
    fprintf(stderr, "cannot read %s\n", o.input.c_str());
    return false;
  }
  audio_tools::AudioInfo target(16000, 1, 16);
  bool is_wav = bytes.size() > 12 && memcmp(bytes.data(), "RIFF", 4) == 0;
  if (!is_wav) {  // raw 16 kHz mono s16le
    for (size_t p = 0; p < bytes.size(); p += 4096) out.write(bytes.data() + p, std::min<size_t>(4096, bytes.size() - p));
    return true;
  }
  VectorPrint pcm;
  audio_tools::WAVDecoder wav;
  audio_tools::EncodedAudioStream dec(&pcm, &wav);
  dec.begin();
  for (size_t p = 0; p < bytes.size(); p += 4096) dec.write(bytes.data() + p, std::min<size_t>(4096, bytes.size() - p));
  audio_tools::AudioInfo info = wav.audioInfo();
  if (info.bits_per_sample != 16) {
    fprintf(stderr, "only 16 bit WAV files are supported\n");
    return false;
  }
  if (o.verbose)
    fprintf(stderr, "input: %d Hz, %d channels, %zu PCM bytes\n", (int)info.sample_rate, info.channels, pcm.data.size());
  if (info == target) {
    for (size_t p = 0; p < pcm.data.size(); p += 4096)
      out.write(pcm.data.data() + p, std::min<size_t>(4096, pcm.data.size() - p));
    return true;
  }
  audio_tools::FormatConverterStream conv(out);
  conv.begin(info, target);
  for (size_t p = 0; p < pcm.data.size(); p += 4096)
    conv.write(pcm.data.data() + p, std::min<size_t>(4096, pcm.data.size() - p));
  return true;
}

int main(int argc, char** argv) {
  if (!parse(argc, argv, g_opt)) {
    usage();
    return 2;
  }
  const Options& o = g_opt;
  static FilePrint err(stderr), out(stdout);
  audio_tools::AudioToolsLogger.begin(err, o.verbose ? audio_tools::AudioLogger::Info : audio_tools::AudioLogger::Error);

  std::vector<uint8_t> model_bytes;
  ModelContainer mc;
  if (!o.model.empty()) {
    model_bytes = readFile(o.model);
    if (mc.begin(model_bytes.data(), model_bytes.size()) != ModelContainer::OK) {
      fprintf(stderr, "invalid model container %s\n", o.model.c_str());
      return 1;
    }
  }

  PhonemeQueue queue(256);

  // ---- stage 1 only: audio -> phoneme frames on stdout ----
  if (o.phonemes_only) {
    audio_tools::AudioRealFFT fft;
    SpeechToPhonemes stp(fft, queue);
    if (!stp.begin(mc)) {
      fprintf(stderr, "no acoustic model in %s\n", o.model.c_str());
      return 1;
    }
    stp.setBlankThreshold(o.blank_threshold);
    stp.setVADEnabled(!o.no_vad);
    QueueToStream sender(queue, out);
    sender.setTextMode(o.text_phonemes);
    struct Sink : public Print {
      SpeechToPhonemes* s;
      QueueToStream* q;
      size_t write(uint8_t b) override { return write(&b, 1); }
      size_t write(const uint8_t* d, size_t n) override {
        s->write(d, n);
        q->copy();
        return n;
      }
    } sink;
    sink.s = &stp;
    sink.q = &sender;
    if (!feedAudio(o, sink)) return 1;
    stp.endOfUtterance();
    sender.copy();
    return 0;
  }

  // ---- stage 2 setup (shared) ----
  auto setupDecoder = [&](PhonemesToText& ptt) -> bool {
    if (!o.model.empty()) ptt.begin(mc); else ptt.begin();
    ptt.setResultCallback(printResult);
    ptt.decoderConfig().nbest = o.nbest;
    ptt.decoderConfig().beam_size = o.beam;
    if (o.lm_weight >= 0) ptt.decoderConfig().lm_weight = o.lm_weight;
    if (o.word_end_beam >= 0) ptt.decoderConfig().word_end_beam = o.word_end_beam;
    if (o.word_bonus > -1000.0f) ptt.decoderConfig().word_bonus = o.word_bonus;
    ptt.decoderConfig().min_confidence = o.min_confidence;
    ptt.decoderConfig().partial_results = o.verbose;
    for (auto& h : o.hotwords) ptt.addHotword(h.first.c_str(), h.second);
    if (!o.words.empty()) {
      ptt.setWordListMode(true);
      size_t start = 0;
      while (start <= o.words.size()) {
        size_t c = o.words.find(',', start);
        std::string w = o.words.substr(start, c == std::string::npos ? std::string::npos : c - start);
        if (!w.empty()) ptt.addWord(w.c_str());
        if (c == std::string::npos) break;
        start = c + 1;
      }
    }
    if (!o.grammar.empty() && !ptt.setGrammar(o.grammar.c_str())) {
      fprintf(stderr, "grammar error: %s\n", ptt.grammar().error().c_str());
      return false;
    }
    return true;
  };

  // ---- stage 2 only: phoneme frames from stdin -> text ----
  if (o.from_phonemes) {
    PhonemesToText ptt(queue);
    if (!setupDecoder(ptt)) return 1;
    StreamToQueue receiver;
    receiver.queueWriter().setQueue(queue);
    uint8_t buf[256];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) {
      receiver.write(buf, n);
      ptt.copy();
    }
    ptt.copy();
    return 0;
  }

  // ---- both stages ----
  audio_tools::AudioRealFFT fft;
  STTPipeline stt(fft, 256);
  if (!stt.speechToPhonemes().begin(mc)) {
    fprintf(stderr, "no acoustic model in %s\n", o.model.c_str());
    return 1;
  }
  if (!setupDecoder(stt.phonemesToText())) return 1;
  stt.speechToPhonemes().setBlankThreshold(o.blank_threshold);
  stt.speechToPhonemes().setVADEnabled(!o.no_vad);

#ifdef TINYSTT_HAVE_MIC
  if (o.mic) {
    audio_tools::MiniAudioStream mic;
    auto cfg = mic.defaultConfig(audio_tools::RX_MODE);
    cfg.sample_rate = 16000;
    cfg.channels = 1;
    cfg.bits_per_sample = 16;
    mic.begin(cfg);
    fprintf(stderr, "listening (Ctrl-C to stop)...\n");
    uint8_t buf[640];
    while (true) {
      size_t n = mic.readBytes(buf, sizeof(buf));
      if (n > 0) stt.write(buf, n);
      stt.copy();
    }
  }
#else
  if (o.mic) {
    fprintf(stderr, "built without microphone support (-DTINYSTT_DESKTOP_MIC=ON)\n");
    return 1;
  }
#endif

  struct Sink : public Print {
    STTPipeline* s;
    size_t write(uint8_t b) override { return write(&b, 1); }
    size_t write(const uint8_t* d, size_t n) override {
      s->write(d, n);
      s->copy();
      return n;
    }
  } sink;
  sink.s = &stt;
  if (!feedAudio(o, sink)) return 1;
  stt.endOfUtterance();
  stt.copy();
  return 0;
}
