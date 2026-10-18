#pragma once
#include "AudioTools.h"
#include "AudioTools/Concurrency/LockFree/RingBufferSPSC.h"
#include "TinySTT/PhonemeFrame.h"

namespace tinystt {

/// The queue type connecting the two stages: lock-free single producer /
/// single consumer, identical on desktop, emulator and microcontroller.
using PhonemeQueue = audio_tools::RingBufferSPSC<PhonemeFrame>;

/**
 * @brief Producer side helper: writes PhonemeFrames to a queue without ever
 * blocking forever. RingBufferSPSC does not block, so when it is full we retry
 * with delay(1) up to `timeout_ms`, then drop the frame and count it.
 * EndOfUtterance and Reset frames are never dropped (they wait for room).
 */
class PhonemeQueueWriter {
 public:
  PhonemeQueueWriter() = default;
  explicit PhonemeQueueWriter(audio_tools::BaseBuffer<PhonemeFrame>& queue) {
    setQueue(queue);
  }

  void setQueue(audio_tools::BaseBuffer<PhonemeFrame>& queue) { p_queue = &queue; }
  audio_tools::BaseBuffer<PhonemeFrame>* queue() { return p_queue; }

  /// Max wait for room before a Frame/BlankRun is dropped (0: drop at once)
  void setTimeoutMs(uint32_t ms) { timeout_ms = ms; }

  /// Returns false if the frame was dropped
  bool write(const PhonemeFrame& frame) {
    if (p_queue == nullptr) return false;
    bool must_deliver = frame.type == PhonemeFrame::EndOfUtterance ||
                        frame.type == PhonemeFrame::Reset;
    uint32_t start = millis();
    while (!p_queue->write(frame)) {
      if (!must_deliver && (uint32_t)(millis() - start) >= timeout_ms) {
        dropped++;
        LOGW("TinySTT: phoneme queue full - frame %u dropped", (unsigned)frame.index);
        return false;
      }
      delay(1);
    }
    written++;
    return true;
  }

  uint32_t droppedFrames() const { return dropped; }
  uint32_t writtenFrames() const { return written; }
  void resetCounters() { dropped = written = 0; }

 protected:
  audio_tools::BaseBuffer<PhonemeFrame>* p_queue = nullptr;
  uint32_t timeout_ms = 100;
  uint32_t dropped = 0;
  uint32_t written = 0;
};

}  // namespace tinystt
