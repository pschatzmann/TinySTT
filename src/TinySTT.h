#pragma once
/**
 * TinySTT: small speech-to-text for microcontrollers, split into
 *   SpeechToPhonemes (audio -> phoneme frames) and
 *   PhonemesToText   (phoneme frames -> text),
 * connected by a lock-free SPSC queue (same task, two tasks, or two devices
 * via QueueToStream / StreamToQueue). See README.md and docs/.
 */
#include "AudioTools.h"
#include "TinySTT/ModelPartition.h"
#include "TinySTT/PhonemeOutput.h"
#include "TinySTT/STTPipeline.h"
#include "TinySTT/QueueBridge.h"

namespace tinystt {

/**
 * @brief Speech to text as an audio-tools AudioOutput: write 16 kHz mono
 * 16 bit PCM into it (e.g. with a StreamCopy) and it prints each accepted
 * final result as one line to the output Print.
 *
 *   I2SStream mic;  AudioRealFFT fft;  ModelContainer model;
 *   TinySTT stt(fft, model, Serial);
 *   StreamCopy copier(stt, mic);
 *   void setup() { ...; stt.begin(); }   // load the model before begin()
 *   void loop() { copier.copy(); }
 *
 * write() runs SpeechToPhonemes (VAD, features, acoustic model) in the
 * caller's task; begin() starts a task for PhonemesToText (the decoder),
 * pinned to core 0 on ESP32 (Arduino's loop() runs on core 1). The two are
 * connected by a lock-free SPSC queue. On platforms without audio-tools tasks
 * (e.g. RP2040/RP2350 without the core's FreeRTOS option) write() also runs
 * the decoder.
 *
 * Other formats: put a FormatConverterStream / ResampleStream in front.
 * Optional settings (grammar, word list, hotwords, VAD, decoder, batching)
 * go through speechToPhonemes() / phonemesToText() before begin(). For one
 * loop without a task use STTPipeline.
 */
class TinySTT : public audio_tools::AudioOutput {
 public:
  TinySTT(audio_tools::AudioFFTBase& fft, const ModelContainer& model, Print& output, size_t queueSize = 64)
      : p_model(&model), p_output(&output), pipeline(fft, queueSize) {
    // 4 output frames per model pass: much less memory traffic, +90 ms latency
    pipeline.speechToPhonemes().setBatchFrames(4);
    audio_tools::AudioOutput::setAudioInfo(audio_tools::AudioInfo(16000, 1, 16));
  }

  ~TinySTT() { end(); }

  /// Starts recognition: initializes both stages, starts the decoder task
  bool begin() override {
    end();
    if (!*p_model) {
      LOGE("TinySTT: model container not loaded");
      return false;
    }
    if (!pipeline.begin(*p_model)) return false;
    pipeline.phonemesToText().setOutput(*p_output);
#if defined(TINYSTT_HAS_TASKS)
    is_active = pipeline.startDecoderTask(stack_size, priority, decoder_core);
#else
    is_active = true;  // no tasks on this platform: write() also decodes
#endif
    return is_active;
  }

  /// Ends the current utterance and stops the decoder task
  void end() override {
    if (!is_active) return;
    pipeline.endOfUtterance();
#if defined(TINYSTT_HAS_TASKS)
    // let the decoder take the remaining frames (and print the last result)
    uint32_t start = millis();
    while (pipeline.phonemeQueue().available() > 0 && millis() - start < 1000) delay(1);
    pipeline.stopTasks();
#else
    pipeline.copy();
#endif
    is_active = false;
  }

  /// 16 kHz mono int16 PCM; runs speech -> phonemes in the caller's task
  size_t write(const uint8_t* data, size_t len) override {
    if (!is_active) return 0;
    size_t result = pipeline.write(data, len);
#if !defined(TINYSTT_HAS_TASKS)
    pipeline.copy();  // decode in the caller's task
#endif
    return result;
  }

  /// Only 16 kHz, mono, 16 bit is supported (convert in front otherwise)
  void setAudioInfo(audio_tools::AudioInfo info) override {
    if (info.sample_rate != 16000 || info.channels != 1 || info.bits_per_sample != 16) {
      LOGE("TinySTT: needs 16000 Hz, 1 channel, 16 bit (got %d Hz, %d ch, %d bit) - use a FormatConverterStream",
           (int)info.sample_rate, info.channels, info.bits_per_sample);
      return;
    }
    audio_tools::AudioOutput::setAudioInfo(info);
  }

  /// Ends the current utterance now (normally the VAD does this)
  void endOfUtterance() { pipeline.endOfUtterance(); }

  bool isActive() const { return is_active; }

  /// Decoder task settings, used by begin() (stack in bytes; core -1: any).
  /// The default core is the one loop() does not run on: 0 on ESP32, 1 on
  /// the RP2040/RP2350.
  void setTaskConfig(int stackSize, int taskPriority = 1, int decoderCore = kDefaultDecoderCore) {
    stack_size = stackSize;
    priority = taskPriority;
    decoder_core = decoderCore;
  }

  /// Stage 1 settings (VAD, batching, top-k, blank skipping); before begin()
  SpeechToPhonemes& speechToPhonemes() { return pipeline.speechToPhonemes(); }
  /// Stage 2 settings (grammar, word list, hotwords, decoder, an additional
  /// result callback); before begin()
  PhonemesToText& phonemesToText() { return pipeline.phonemesToText(); }
  /// The last final result (also when it was not accepted)
  const SttResult& lastResult() { return pipeline.phonemesToText().lastResult(); }

 protected:
#if defined(ARDUINO_ARCH_RP2040)
  static constexpr int kDefaultDecoderCore = 1;  // loop() runs on core 0
#else
  static constexpr int kDefaultDecoderCore = 0;  // ESP32: loop() runs on core 1
#endif
  const ModelContainer* p_model;
  Print* p_output;
  STTPipeline pipeline;
  bool is_active = false;
  int stack_size = 16 * 1024;
  int priority = 1;
  int decoder_core = kDefaultDecoderCore;
};

}  // namespace tinystt

#ifndef TINYSTT_NO_USING_NAMESPACE
using namespace tinystt;
#endif
