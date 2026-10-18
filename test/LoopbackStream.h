#pragma once
// Byte loopback Stream for tests: what is written can be read back.
// max_chunk limits available()/readBytes() to simulate fragmented transport.
#include <deque>

#include "AudioTools.h"

class LoopbackStream : public Stream {
 public:
  explicit LoopbackStream(int maxChunk = 0) : max_chunk(maxChunk) {}
  size_t write(uint8_t b) override {
    data.push_back(b);
    return 1;
  }
  size_t write(const uint8_t* buf, size_t len) override {
    data.insert(data.end(), buf, buf + len);
    return len;
  }
  int available() override {
    int n = (int)data.size();
    return (max_chunk > 0 && n > max_chunk) ? max_chunk : n;
  }
  int read() override {
    if (data.empty()) return -1;
    int b = data.front();
    data.pop_front();
    return b;
  }
  int peek() override { return data.empty() ? -1 : data.front(); }
  void flush() override {}
  std::deque<uint8_t> data;
  int max_chunk = 0;
};
