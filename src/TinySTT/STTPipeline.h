#pragma once
#include <atomic>
#include <functional>

#include "AudioTools.h"
#if defined(ARDUINO_ARCH_RP2040) && defined(__FREERTOS)
// RP2040/RP2350 core with the FreeRTOS option: must be included before the
// audio-tools concurrency header so that it provides Task
#include <FreeRTOS.h>
#endif
#include "AudioTools/Concurrency/Concurrency.h"

// audio-tools Task: ESP32 (FreeRTOS), desktop (std::thread), RP2040/RP2350
// with FreeRTOS. Without it TinySTT decodes in the caller's task.
#if defined(ESP32) || defined(USE_CPP_TASK) || defined(USE_FREERTOS)
#define TINYSTT_HAS_TASKS 1
#endif
#include "TinySTT/PhonemeQueue.h"
#include "TinySTT/PhonemesToText.h"
#include "TinySTT/SpeechToPhonemes.h"

namespace tinystt {

/**
 * @brief Both stages on one device, connected by one lock-free SPSC queue
 * (RingBufferSPSC<PhonemeFrame>).
 *
 * Single task: call write() with PCM and copy() in the same loop.
 * Two tasks: startTasks(audioSource) runs SpeechToPhonemes on one task
 * (reading the audio source) and PhonemesToText on another; on ESP32 they are
 * pinned to different cores.
 */
class STTPipeline {
 public:
  explicit STTPipeline(audio_tools::AudioFFTBase& fft, size_t queueSize = 64)
      : queue(queueSize), stt(fft, queue), ptt(queue) {}

#if defined(TINYSTT_HAS_TASKS)
  ~STTPipeline() { stopTasks(); }
#endif

  bool begin(const ModelContainer& mc,
             AcousticModel::Precision precision = AcousticModel::Precision::Auto) {
    if (!stt.begin(mc, precision)) return false;
    return ptt.begin(mc);
  }

  /// 16 kHz mono int16 PCM (single task use)
  size_t write(const uint8_t* data, size_t len) { return stt.write(data, len); }
  /// Decodes the queued phonemes (single task use)
  size_t copy() { return ptt.copy(); }
  void endOfUtterance() { stt.endOfUtterance(); }

#if defined(TINYSTT_HAS_TASKS)
  /**
   * Starts the producer task (reads `audio` in chunks into SpeechToPhonemes)
   * and the consumer task (PhonemesToText::copy()).
   */
  bool startTasks(Stream& audio, int stackSize = 16 * 1024, int priority = 1, int audioCore = 1,
                  int decoderCore = 0) {
    p_audio = &audio;
    stop_requested = false;
    bool ok = startTask(producer, "tinystt-audio", stackSize, priority, audioCore, producer_parked, [this] {
      uint8_t buf[640];
      int avail = p_audio->available();
      if (avail <= 0) {
        delay(1);
        return;
      }
      size_t n = p_audio->readBytes(buf, avail < (int)sizeof(buf) ? avail : (int)sizeof(buf));
      if (n > 0) stt.write(buf, n);
    });
    tasks_running = ok;
    ok = ok && startDecoderTask(stackSize, priority, decoderCore);
    return ok;
  }

  /**
   * Starts only the consumer task (PhonemesToText::copy()); the caller feeds
   * PCM with write() from its own task (e.g. a StreamCopy in loop()).
   */
  bool startDecoderTask(int stackSize = 16 * 1024, int priority = 1, int core = 0) {
    stop_requested = false;
    decoder_running = startTask(consumer, "tinystt-decoder", stackSize, priority, core, consumer_parked, [this] {
      if (ptt.copy(kDecoderMessages) == 0) delay(1);
    });
    return decoder_running;
  }

  /**
   * Stops the tasks cooperatively: they finish their current step and park
   * before they are suspended, so a task is never stopped in the middle of
   * an operation (e.g. holding the heap lock inside malloc).
   */
  void stopTasks() {
    if (!tasks_running && !decoder_running) return;
    stop_requested = true;
    uint32_t start = millis();
    while ((tasks_running && !producer_parked) || (decoder_running && !consumer_parked)) {
      if (millis() - start > 2000) {
        LOGW("TinySTT: tasks did not park within 2 s");
        break;
      }
      delay(1);
    }
    if (tasks_running) suspendTask(producer);
    if (decoder_running) suspendTask(consumer);
    tasks_running = decoder_running = false;
  }

#endif

  SpeechToPhonemes& speechToPhonemes() { return stt; }
  PhonemesToText& phonemesToText() { return ptt; }
  PhonemeQueue& phonemeQueue() { return queue; }

 protected:
  PhonemeQueue queue;
  SpeechToPhonemes stt;
  PhonemesToText ptt;
#if defined(TINYSTT_HAS_TASKS)
  audio_tools::Task producer, consumer;
  Stream* p_audio = nullptr;
  bool tasks_running = false;
  bool decoder_running = false;
  static constexpr uint32_t kMaxBusyMs = 50;  ///< longest time a task runs without yielding
  static constexpr size_t kDecoderMessages = 16;  ///< queue messages per decoder task step
  std::atomic<bool> stop_requested{false};
  std::atomic<bool> producer_parked{false};
  std::atomic<bool> consumer_parked{false};

  /// Suspends the task and waits until it really is suspended: suspending a
  /// task that runs on the other core can return before that core switched
  /// it out, and Task::begin() then replaces the std::function the task may
  /// still be calling (crash on the ESP32-P4).
  static void suspendTask(audio_tools::Task& task) {
    task.end();
#if defined(ESP32) || (defined(USE_FREERTOS) && !defined(USE_CPP_TASK))
    TaskHandle_t h = task.getTaskHandle();
    uint32_t start = millis();
    while (h != nullptr && eTaskGetState(h) != eSuspended && millis() - start < 1000) delay(1);
#endif
  }

  /// Creates the task if it does not exist yet (an existing, stopped task is
  /// resumed) and runs `step` until stopTasks()
  bool startTask(audio_tools::Task& task, const char* name, int stackSize, int priority, int core,
                 std::atomic<bool>& parked, std::function<void()> step) {
    parked = false;
#if !defined(ESP32) && !defined(USE_CPP_TASK)
    // FreeRTOS outside ESP-IDF takes the stack depth in words, not bytes
    stackSize /= (int)sizeof(StackType_t);
#endif
    if (task.isEnded()) {
      if (!task.create(name, stackSize, priority, core)) return false;
      suspendTask(task);  // create() starts the task and then suspends it
    }
    uint32_t last_yield = millis();
    return task.begin([this, &parked, step, last_yield]() mutable {
      if (stop_requested) {  // park at a safe point until suspended
        parked = true;
        delay(5);
        return;
      }
      parked = false;
      step();
      // a task that always has work (e.g. a decoder slower than real time)
      // must still let the idle task run, or the task watchdog resets
      if (millis() - last_yield >= kMaxBusyMs) {
        delay(1);
        last_yield = millis();
      }
    });
  }
#endif
};

}  // namespace tinystt
