#pragma once
#include "TinySTT/PhonemeProtocol.h"
#include "TinySTT/PhonemeQueue.h"

namespace tinystt {

/**
 * @brief Sender side of a split setup: the consumer of the local phoneme
 * queue. Serializes every queued PhonemeFrame onto any Arduino Print/Stream
 * (Serial, UDP, ESP-NOW wrapper, stdout ...). Call copy() from the task loop.
 */
class QueueToStream {
 public:
  QueueToStream() = default;
  QueueToStream(audio_tools::BaseBuffer<PhonemeFrame>& queue, Print& out) {
    begin(queue, out);
  }

  void begin(audio_tools::BaseBuffer<PhonemeFrame>& queue, Print& out) {
    p_queue = &queue;
    p_out = &out;
    protocol.reset();
  }

  /// Text mode prints ARPAbet lines instead of binary frames (debugging)
  void setTextMode(bool flag) { text_mode = flag; }

  /// Sends all queued frames; returns the number of frames sent
  size_t copy() {
    if (p_queue == nullptr || p_out == nullptr) return 0;
    size_t count = 0;
    PhonemeFrame frame;
    while (p_queue->read(frame)) {
      if (text_mode)
        PhonemeProtocol::printText(*p_out, frame);
      else
        protocol.write(*p_out, frame);
      count++;
    }
    return count;
  }

 protected:
  audio_tools::BaseBuffer<PhonemeFrame>* p_queue = nullptr;
  Print* p_out = nullptr;
  PhonemeProtocol protocol;
  bool text_mode = false;
};

/**
 * @brief Receiver side of a split setup: the producer of the local phoneme
 * queue. Parses PhonemeProtocol bytes from any Arduino Stream and writes the
 * frames into the queue that PhonemesToText consumes.
 */
class StreamToQueue {
 public:
  StreamToQueue() = default;
  StreamToQueue(Stream& in, audio_tools::BaseBuffer<PhonemeFrame>& queue) {
    begin(in, queue);
  }

  void begin(Stream& in, audio_tools::BaseBuffer<PhonemeFrame>& queue) {
    p_in = &in;
    writer.setQueue(queue);
    protocol.reset();
  }

  PhonemeQueueWriter& queueWriter() { return writer; }
  PhonemeProtocol& protocolState() { return protocol; }

  /// Processes the available bytes; returns the number of frames queued
  size_t copy() {
    if (p_in == nullptr) return 0;
    size_t count = 0;
    uint8_t buf[64];
    int avail;
    while ((avail = p_in->available()) > 0) {
      int n = p_in->readBytes(buf, avail < (int)sizeof(buf) ? avail : (int)sizeof(buf));
      if (n <= 0) break;
      count += write(buf, n);
    }
    return count;
  }

  /// Feeds raw bytes directly (e.g. from an ESP-NOW receive callback)
  size_t write(const uint8_t* data, size_t len) {
    size_t count = 0;
    PhonemeFrame frame;
    for (size_t j = 0; j < len; j++) {
      protocol.feed(data[j]);
      while (protocol.next(frame)) {
        if (writer.write(frame)) count++;
      }
    }
    return count;
  }

 protected:
  Stream* p_in = nullptr;
  PhonemeQueueWriter writer;
  PhonemeProtocol protocol;
};

}  // namespace tinystt
