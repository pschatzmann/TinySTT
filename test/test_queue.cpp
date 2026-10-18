// Step 2: phoneme table, PhonemeFrame, the SPSC queue between the stages,
// and the stream bridge with its wire protocol.
#include <atomic>
#include <thread>

#include "AudioTools.h"
#include "AudioTools/Concurrency/Concurrency.h"
#include "LoopbackStream.h"
#include "TestUtil.h"
#include "TinySTT/PhonemeOutput.h"
#include "TinySTT/QueueBridge.h"

using namespace tinystt;

static PhonemeFrame makeFrame(uint32_t idx) {
  PhonemeFrame f;
  f.index = idx;
  f.k = 1 + idx % PhonemeFrame::kMaxTopK;
  for (int j = 0; j < f.k; j++) {
    f.id[j] = (uint8_t)((idx + j) % Phonemes::kCount);
    f.logprob[j] = PhonemeFrame::quantize(-0.1f * (j + 1) - (idx % 7) * 0.5f);
  }
  return f;
}

// ---- phonemes / frame ----

TEST(phoneme_table) {
  CHECK_EQ(Phonemes::id("AA"), 2);
  CHECK_EQ(Phonemes::id("zh"), 40);
  CHECK_EQ(Phonemes::id("AH1"), Phonemes::id("AH"));
  CHECK_EQ(Phonemes::id("sil"), Phonemes::kSilence);
  CHECK_EQ(Phonemes::id("XX"), Phonemes::kInvalid);
  CHECK_EQ(Phonemes::id("AHHHHHHHHH"), Phonemes::kInvalid);
  for (int j = 0; j < Phonemes::kCount; j++) CHECK_EQ(Phonemes::id(Phonemes::name(j)), j);
  CHECK(Phonemes::isVowel(Phonemes::id("OW")));
  CHECK(Phonemes::isVowel(Phonemes::id("UW")));
  CHECK(Phonemes::isVowel(Phonemes::id("IY")));
  CHECK(!Phonemes::isVowel(Phonemes::id("P")));
  CHECK(!Phonemes::isVowel(Phonemes::id("OY") + 1));
  int vowels = 0;
  for (int j = 0; j < Phonemes::kCount; j++) vowels += Phonemes::isVowel(j);
  CHECK_EQ(vowels, 15);
}

TEST(frame_quantization) {
  CHECK_EQ(PhonemeFrame::quantize(0.0f), 0);
  CHECK_EQ(PhonemeFrame::quantize(0.5f), 0);  // clamped to <= 0
  CHECK_EQ(PhonemeFrame::quantize(-100.0f), -128);
  for (float lp = -7.9f; lp <= 0; lp += 0.37f)
    CHECK_NEAR(PhonemeFrame::dequantize(PhonemeFrame::quantize(lp)), lp, 0.5 / PhonemeFrame::kLogProbScale + 1e-6);
  PhonemeFrame f = makeFrame(3);
  CHECK_NEAR(f.logProbOf(f.id[0]), f.logProb(0), 0);
  CHECK(f.logProbOf(39) < f.logProb(f.k - 1));
  PhonemeFrame b = PhonemeFrame::blankRun(10, 5);
  CHECK_NEAR(b.logProbOf(Phonemes::kBlank), 0, 0);
}

// ---- SPSC queue ----

TEST(queue_order_and_eou) {
  PhonemeQueue q(64);
  PhonemeQueueWriter w(q);
  for (uint32_t j = 0; j < 40; j++) CHECK(w.write(makeFrame(j)));
  CHECK(w.write(PhonemeFrame::endOfUtterance(40)));
  PhonemeFrame f;
  for (uint32_t j = 0; j < 40; j++) {
    CHECK(q.read(f));
    CHECK(f == makeFrame(j));
  }
  CHECK(q.read(f));
  CHECK_EQ(f.type, PhonemeFrame::EndOfUtterance);
  CHECK(!q.read(f));
}

TEST(queue_full_drops_after_timeout) {
  PhonemeQueue q(4);
  PhonemeQueueWriter w(q);
  w.setTimeoutMs(5);
  int capacity = q.size();
  for (int j = 0; j < capacity; j++) CHECK(w.write(makeFrame(j)));
  uint32_t start = millis();
  CHECK(!w.write(makeFrame(99)));
  CHECK((millis() - start) >= 5);
  CHECK_EQ(w.droppedFrames(), 1u);
  CHECK_EQ(w.writtenFrames(), (uint32_t)capacity);
}

TEST(queue_eou_never_dropped) {
  PhonemeQueue q(4);
  PhonemeQueueWriter w(q);
  w.setTimeoutMs(0);
  int capacity = q.size();
  for (int j = 0; j < capacity; j++) w.write(makeFrame(j));
  std::thread consumer([&] {
    delay(30);
    PhonemeFrame f;
    q.read(f);
  });
  CHECK(w.write(PhonemeFrame::endOfUtterance(7)));  // waits for the consumer
  consumer.join();
  CHECK_EQ(w.droppedFrames(), 0u);
}

TEST(queue_stress_two_tasks) {
  const uint32_t kFrames = 100000;
  PhonemeQueue q(64);
  PhonemeQueueWriter w(q);
  w.setTimeoutMs(1000);
  std::atomic<uint32_t> produced{0}, consumed{0};
  std::atomic<bool> order_ok{true};
  audio_tools::Task producer("producer", 4096, 1, 0);
  audio_tools::Task consumer("consumer", 4096, 1, 1);
  producer.begin([&] {
    uint32_t j = produced.load();
    if (j < kFrames) {
      w.write(makeFrame(j));
      produced.store(j + 1);
    } else {
      delay(1);
    }
  });
  consumer.begin([&] {
    PhonemeFrame f;
    if (q.read(f)) {
      uint32_t j = consumed.load();
      if (f != makeFrame(j)) order_ok = false;
      consumed.store(j + 1);
    }
  });
  uint32_t start = millis();
  while (consumed.load() < kFrames && millis() - start < 20000) delay(5);
  producer.end();
  consumer.end();
  CHECK_EQ(consumed.load(), kFrames);
  CHECK(order_ok.load());
  CHECK_EQ(w.droppedFrames(), 0u);
}

// ---- protocol / bridge ----

TEST(protocol_round_trip_all_types) {
  PhonemeProtocol tx, rx;
  std::vector<PhonemeFrame> frames = {makeFrame(0), makeFrame(1), makeFrame(2), makeFrame(3),
                                      PhonemeFrame::blankRun(4, 300),
                                      PhonemeFrame::endOfUtterance(304), PhonemeFrame::reset(),
                                      makeFrame(0xFFFFFFF0u)};
  for (auto& f : frames) {
    uint8_t buf[PhonemeProtocol::kMaxFrameSize];
    int n = tx.encode(f, buf);
    CHECK(n <= PhonemeProtocol::kMaxFrameSize);
    PhonemeFrame out;
    int got = 0;
    for (int j = 0; j < n; j++) got += rx.decode(buf[j], out);
    CHECK_EQ(got, 1);
    CHECK(out == f);
  }
  CHECK_EQ(rx.crcErrors(), 0u);
  CHECK_EQ(rx.lostFrames(), 0u);
}

TEST(bridge_round_trip_chunked) {
  for (int chunk : {0, 1, 3, 7}) {
    PhonemeQueue q_tx(64), q_rx(64);
    LoopbackStream link(chunk);
    QueueToStream sender(q_tx, link);
    StreamToQueue receiver(link, q_rx);
    PhonemeQueueWriter w(q_tx);
    for (uint32_t j = 0; j < 50; j++) w.write(makeFrame(j));
    w.write(PhonemeFrame::endOfUtterance(50));
    CHECK_EQ(sender.copy(), 51u);
    size_t received = 0;
    while (link.available()) received += receiver.copy();
    CHECK_EQ(received, 51u);
    PhonemeFrame f;
    for (uint32_t j = 0; j < 50; j++) {
      CHECK(q_rx.read(f));
      CHECK(f == makeFrame(j));
    }
    CHECK(q_rx.read(f));
    CHECK_EQ(f.type, PhonemeFrame::EndOfUtterance);
  }
}

TEST(protocol_resync_after_corruption) {
  PhonemeProtocol tx, rx;
  std::vector<uint8_t> wire;
  uint8_t buf[PhonemeProtocol::kMaxFrameSize];
  for (uint32_t j = 0; j < 20; j++) {
    int n = tx.encode(makeFrame(j), buf);
    wire.insert(wire.end(), buf, buf + n);
    if (j == 5) {  // garbage between frames, including fake sync bytes
      uint8_t junk[] = {0x00, 0xA5, 0xFF, 0xA5, 0x01, 0x13};
      wire.insert(wire.end(), junk, junk + sizeof(junk));
    }
  }
  // corrupt one byte inside frame 10 (find its sync by re-encoding offsets)
  size_t pos = 0;
  PhonemeProtocol sizer;
  for (uint32_t j = 0; j < 10; j++) pos += sizer.encode(makeFrame(j), buf) + (j == 5 ? 6 : 0);
  wire[pos + 4] ^= 0x40;

  std::vector<PhonemeFrame> got;
  PhonemeFrame f;
  for (uint8_t b : wire) {
    rx.feed(b);
    while (rx.next(f)) got.push_back(f);
  }
  // every frame except the corrupted one survives, in order
  CHECK_EQ(got.size(), 19u);
  uint32_t expected = 0;
  for (auto& g : got) {
    if (expected == 10) expected++;
    CHECK(g == makeFrame(expected));
    expected++;
  }
  CHECK(rx.crcErrors() >= 1u);
  CHECK_EQ(rx.lostFrames(), 1u);  // detected via the sequence counter
}

TEST(protocol_frame_hidden_in_corrupt_frame) {
  // A truncated k=4 frame followed directly by a complete k=0 frame: the
  // parser has to find the second frame inside the bytes of the first.
  PhonemeProtocol tx, rx;
  uint8_t a[PhonemeProtocol::kMaxFrameSize], b[PhonemeProtocol::kMaxFrameSize];
  PhonemeFrame big = makeFrame(3);  // k = 4
  int na = tx.encode(big, a);
  PhonemeFrame small = PhonemeFrame::endOfUtterance(9);
  int nb = tx.encode(small, b);
  std::vector<uint8_t> wire(a, a + 12);  // header + 1 entry only
  wire.insert(wire.end(), b, b + nb);
  (void)na;
  std::vector<PhonemeFrame> got;
  PhonemeFrame f;
  for (uint8_t x : wire) {
    rx.feed(x);
    while (rx.next(f)) got.push_back(f);
  }
  CHECK_EQ(got.size(), 1u);
  if (!got.empty()) CHECK(got[0] == small);
}

TEST(bridge_text_mode) {
  PhonemeQueue q(8);
  PhonemeFrame f;
  f.index = 12;
  f.k = 2;
  f.id[0] = Phonemes::id("HH");
  f.logprob[0] = PhonemeFrame::quantize(-0.25f);
  f.id[1] = Phonemes::id("AH");
  f.logprob[1] = PhonemeFrame::quantize(-1.75f);
  q.write(f);
  q.write(PhonemeFrame::blankRun(13, 4));
  LoopbackStream out;
  QueueToStream s(q, out);
  s.setTextMode(true);
  s.copy();
  std::string text(out.data.begin(), out.data.end());
  CHECK_STR(text, "#12 HH:-0.25 AH:-1.75\n#13 <blank x4>\n");
}

// ---- PhonemeOutput ----

static PhonemeFrame top1(uint32_t idx, const char* phone, float lp = -0.1f) {
  PhonemeFrame f;
  f.index = idx;
  f.k = 1;
  f.id[0] = Phonemes::id(phone);
  f.logprob[0] = PhonemeFrame::quantize(lp);
  return f;
}

static std::string withoutCR(const std::deque<uint8_t>& d) {
  std::string s;
  for (uint8_t c : d)
    if (c != '\r') s += (char)c;
  return s;
}

TEST(phoneme_output_phonemes_mode) {
  LoopbackStream out;
  PhonemeOutput po(out);
  CHECK(po.mode() == PhonemeOutput::Phonemes);
  // HH HH <b> AH L L <blank run> L OW sil <eou>  ->  "HH AH L L OW"
  po.write(top1(0, "HH"));
  po.write(top1(1, "HH"));  // repeat collapses
  po.write(top1(2, "<b>"));
  po.write(top1(3, "AH"));
  po.write(top1(4, "L"));
  po.write(top1(5, "L"));
  po.write(PhonemeFrame::blankRun(6, 3));  // separates the repeated L
  po.write(top1(9, "L"));
  po.write(top1(10, "OW"));
  po.write(top1(11, "sil"));
  po.write(PhonemeFrame::endOfUtterance(12));
  // second utterance through the queue interface (writeArray)
  PhonemeFrame second[] = {top1(0, "G"), top1(1, "OW"), PhonemeFrame::endOfUtterance(2)};
  CHECK_EQ(po.writeArray(second, 3), 3);
  // an utterance without phonemes prints nothing
  po.write(PhonemeFrame::blankRun(0, 10));
  po.write(PhonemeFrame::endOfUtterance(10));
  CHECK_STR(withoutCR(out.data), "HH AH L L OW\nG OW\n");
  // nothing is stored
  PhonemeFrame f;
  CHECK(!po.read(f));
  CHECK_EQ(po.available(), 0);
  CHECK(po.availableForWrite() > 0);
}

TEST(phoneme_output_frames_mode) {
  LoopbackStream a, b;
  PhonemeOutput po(a, PhonemeOutput::Frames);
  PhonemeFrame frames[] = {makeFrame(3), PhonemeFrame::blankRun(4, 7), PhonemeFrame::endOfUtterance(11)};
  for (auto& f : frames) {
    po.write(f);
    PhonemeProtocol::printText(b, f);
  }
  CHECK_STR(withoutCR(a.data), withoutCR(b.data));
}

TEST_MAIN()
