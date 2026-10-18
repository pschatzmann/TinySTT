// Step 4: reference ops, the streaming AcousticModel (float32 and int8) against
// numpy goldens (research/model_numpy.py), and FrameBuilder (top-k, blank runs).
#include <cstring>
#include <random>

#include "AudioTools.h"
#include "TestData.h"
#include "TestUtil.h"
#include "TinySTT/AcousticModel.h"
#include "TinySTT/FrameBuilder.h"

using namespace tinystt;

// ---- ops ----

TEST(ops_dense_f32_and_int8) {
  const int R = 3, C = 4;
  float w[R * C] = {1, -2, 0.5f, 0, 0.25f, 0.25f, 0.25f, 0.25f, -1, 0, 0, 3};
  float b[R] = {0.5f, 0, -1};
  float x[C] = {1, 2, 3, 4};
  DenseWeights d;
  d.rows = R;
  d.cols = C;
  d.w = w;
  d.bias = b;
  float y[R];
  d.apply(x, y);
  CHECK_NEAR(y[0], 1 - 4 + 1.5 + 0.5, 1e-6);
  CHECK_NEAR(y[1], 2.5, 1e-6);
  CHECK_NEAR(y[2], -1 + 12 - 1, 1e-6);
  // int8 with per-row scale = max|w|/127
  int8_t q[R * C];
  float s[R];
  for (int r = 0; r < R; r++) {
    float mx = 0;
    for (int c = 0; c < C; c++) mx = std::max(mx, std::fabs(w[r * C + c]));
    s[r] = mx / 127.0f;
    for (int c = 0; c < C; c++) q[r * C + c] = (int8_t)std::lround(w[r * C + c] / s[r]);
  }
  DenseWeights dq;
  dq.rows = R;
  dq.cols = C;
  dq.q = q;
  dq.scale = s;
  dq.bias = b;
  CHECK(dq.isQuantized());
  float yq[R];
  dq.apply(x, yq);
  for (int r = 0; r < R; r++) CHECK_NEAR(yq[r], y[r], 0.05);
  CHECK_EQ(d.params(), (uint32_t)(R * C + R));
}

TEST(ops_dot_s8_variants_identical) {
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> d(-128, 127);
  alignas(16) int8_t a[300], b[300];
  for (int len : {0, 1, 3, 4, 15, 16, 17, 64, 255, 256, 300}) {
    for (int j = 0; j < len; j++) {
      a[j] = (int8_t)d(rng);
      b[j] = (int8_t)d(rng);
    }
    int32_t ref = dotS8Reference(a, b, len);
    CHECK_EQ(dotS8Unrolled(a, b, len), ref);
    CHECK_EQ(dotS8(a, b, len), ref);
  }
  // extreme values: no overflow in int32 for the model widths
  for (int j = 0; j < 256; j++) a[j] = b[j] = -128;
  CHECK_EQ(dotS8(a, b, 256), 256 * 128 * 128);
}

TEST(ops_log_softmax) {
  float x[5] = {1000, 1001, 999, -50, 1000};
  logSoftmaxInPlace(x, 5);
  double sum = 0;
  for (float v : x) sum += std::exp(v);
  CHECK_NEAR(sum, 1.0, 1e-5);
  CHECK(x[1] > x[0] && x[0] > x[2]);
  CHECK_NEAR(x[0], x[4], 0);
}

TEST(ops_gru_step_scalar) {
  // hidden = 1, input = 1: compare against the PyTorch formula by hand
  float wih[3] = {0.5f, -0.3f, 0.8f}, whh[3] = {0.1f, 0.2f, -0.4f};
  float bih[3] = {0.01f, 0.02f, 0.03f}, bhh[3] = {-0.01f, 0.0f, 0.05f};
  DenseWeights ih{3, 1, wih, nullptr, nullptr, bih};
  DenseWeights hh{3, 1, whh, nullptr, nullptr, bhh};
  float x = 0.7f, h = -0.2f, scratch[6];
  float r = sigmoidf(0.5f * x + 0.01f + 0.1f * h - 0.01f);
  float z = sigmoidf(-0.3f * x + 0.02f + 0.2f * h + 0.0f);
  float n = tanhf(0.8f * x + 0.03f + r * (-0.4f * h + 0.05f));
  float expected = (1 - z) * n + z * h;
  gruStep(ih, hh, &x, &h, 1, scratch);
  CHECK_NEAR(h, expected, 1e-6);
}

// ---- acoustic model ----

struct Golden {
  uint32_t T = 0, n_mels = 0, out_frames = 0, classes = 0;
  std::vector<float> feats, lp32, lp8, lp8a;
};

static Golden loadGolden() {
  Golden g;
  auto raw = readTestFile("model_golden.bin");
  if (raw.size() < 16) return g;
  memcpy(&g.T, &raw[0], 4);
  memcpy(&g.n_mels, &raw[4], 4);
  memcpy(&g.out_frames, &raw[8], 4);
  memcpy(&g.classes, &raw[12], 4);
  const float* p = (const float*)&raw[16];
  g.feats.assign(p, p + g.T * g.n_mels);
  p += g.T * g.n_mels;
  g.lp32.assign(p, p + g.out_frames * g.classes);
  p += g.out_frames * g.classes;
  g.lp8.assign(p, p + g.out_frames * g.classes);
  p += g.out_frames * g.classes;
  g.lp8a.assign(p, p + g.out_frames * g.classes);
  return g;
}

static std::vector<float> runModel(AcousticModel& m, const Golden& g) {
  m.reset();
  std::vector<float> out;
  auto take = [&](int n) {
    for (int i = 0; i < n; i++) out.insert(out.end(), m.logProbs(i), m.logProbs(i) + m.classCount());
  };
  for (uint32_t t = 0; t < g.T; t++) take(m.write(&g.feats[t * g.n_mels]));
  take(m.flush());
  return out;
}

static double maxDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return 1e9;
  double d = 0;
  for (size_t j = 0; j < a.size(); j++) d = std::max(d, (double)std::fabs(a[j] - b[j]));
  return d;
}

TEST(model_matches_numpy_f32_and_int8) {
  auto img = readTestFile("placeholder_model.tstt");
  CHECK(!img.empty());
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), img.size()), ModelContainer::OK);
  Golden g = loadGolden();
  CHECK(g.T > 0);

  AcousticModel f32;
  CHECK(f32.begin(mc, AcousticModel::Precision::Float32));
  CHECK(!f32.isQuantized());
  auto out32 = runModel(f32, g);
  CHECK_EQ(out32.size(), (size_t)g.out_frames * g.classes);
  double d32 = maxDiff(out32, g.lp32);
  std::printf("  f32  max diff %g\n", d32);
  CHECK(d32 < 1e-4);

  AcousticModel q8;
  CHECK(q8.begin(mc, AcousticModel::Precision::Int8));
  CHECK(q8.isQuantized());
  double d8 = maxDiff(runModel(q8, g), g.lp8);
  std::printf("  int8 max diff %g (vs numpy int8)\n", d8);
  CHECK(d8 < 1e-4);
  // int8 stays close to float
  double dq = maxDiff(runModel(q8, g), g.lp32);
  std::printf("  int8 vs f32 max diff %g\n", dq);
  CHECK(dq < 0.5);  // quantization error of random weights, not a precision test

  AcousticModel q8a;
  CHECK(q8a.begin(mc, AcousticModel::Precision::Int8Activations));
  CHECK(q8a.usesInt8Activations());
  double d8a = maxDiff(runModel(q8a, g), g.lp8a);
  std::printf("  int8 activations max diff %g (vs numpy)\n", d8a);
  CHECK(d8a < 1e-4);
  double d8af = maxDiff(runModel(q8a, g), g.lp32);
  std::printf("  int8 activations vs f32 max diff %g\n", d8af);
  CHECK(d8af < 0.5);

  AcousticModel autop;
  CHECK(autop.begin(mc));  // Auto: int8 weights + activations when available
  CHECK(autop.isQuantized());
  CHECK(autop.usesInt8Activations());
}

TEST(model_stride_and_streaming) {
  auto img = readTestFile("placeholder_model.tstt");
  ModelContainer mc;
  mc.begin(img.data(), img.size());
  AcousticModel m;
  CHECK(m.begin(mc, AcousticModel::Precision::Float32));
  CHECK_EQ(m.config().conv_stride, 3u);
  Golden g = loadGolden();
  // output count = ceil(T / stride)
  for (uint32_t n : {1u, 2u, 3u, 4u, 30u, 31u}) {
    m.reset();
    uint32_t outs = 0;
    for (uint32_t t = 0; t < n; t++) outs += m.write(&g.feats[(t % g.T) * g.n_mels]);
    outs += m.flush();
    CHECK_EQ(outs, (n + 2) / 3);
    CHECK_EQ(m.outputFrames(), outs);
  }
  // reset() makes runs reproducible (state fully cleared)
  m.reset();
  auto a = runModel(m, g);
  m.reset();
  auto b = runModel(m, g);
  CHECK(a == b);
}

TEST(model_batching_is_exact) {
  auto img = readTestFile("placeholder_model.tstt");
  ModelContainer mc;
  mc.begin(img.data(), img.size());
  Golden g = loadGolden();
  for (auto p : {AcousticModel::Precision::Float32, AcousticModel::Precision::Int8,
                 AcousticModel::Precision::Int8Activations}) {
    AcousticModel ref;
    CHECK(ref.begin(mc, p));
    auto expected = runModel(ref, g);
    for (int b : {2, 3, 4, 5}) {
      AcousticModel m;
      m.setBatchFrames(b);
      CHECK(m.begin(mc, p));
      auto got = runModel(m, g);
      CHECK(got == expected);  // bit identical
      // outputs only arrive in full batches until flush()
      m.reset();
      int first = 0;
      for (uint32_t t = 0; t < 3 * (uint32_t)(b - 1); t++) first += m.write(&g.feats[t * g.n_mels]);
      CHECK_EQ(first, 0);
    }
  }
}

TEST(model_rejects_bad_container) {
  ModelContainerWriter w;
  uint32_t cfg[8] = {1, 80, 32, 5, 3, 2, 32, 41};
  w.addU32("model.config", cfg, 8);
  auto img = w.build();
  ModelContainer mc;
  mc.begin(img.data(), img.size());
  AcousticModel m;
  CHECK(!m.begin(mc));  // weights missing
  CHECK(!m);
  float f[80] = {0};
  CHECK(!m.write(f));
}

// ---- frame builder ----

static std::vector<float> logp(std::initializer_list<std::pair<int, float>> probs) {
  std::vector<float> p(Phonemes::kCount, 1e-6f);
  for (auto& kv : probs) p[kv.first] = kv.second;
  for (auto& v : p) v = logf(v);
  return p;
}

TEST(frame_builder_top_k) {
  PhonemeQueue q(16);
  PhonemeQueueWriter w(q);
  FrameBuilder fb(w);
  fb.setTopK(3);
  fb.setBlankThreshold(1.0f);  // no skipping
  auto p = logp({{5, 0.6f}, {9, 0.3f}, {0, 0.05f}, {12, 0.04f}});
  fb.write(p.data(), Phonemes::kCount);
  PhonemeFrame f;
  CHECK(q.read(f));
  CHECK_EQ(f.type, PhonemeFrame::Frame);
  CHECK_EQ(f.k, 3);
  CHECK_EQ(f.id[0], 5);
  CHECK_EQ(f.id[1], 9);
  CHECK_EQ(f.id[2], 0);
  CHECK_NEAR(f.logProb(0), logf(0.6f), 0.04);
  CHECK(f.logprob[0] >= f.logprob[1] && f.logprob[1] >= f.logprob[2]);
}

TEST(frame_builder_blank_runs) {
  PhonemeQueue q(64);
  PhonemeQueueWriter w(q);
  FrameBuilder fb(w);
  auto blank = logp({{0, 0.99f}});
  auto ah = logp({{Phonemes::id("AH"), 0.9f}, {0, 0.09f}});
  // B B B AH B B AH  then EOU with 2 pending blanks
  for (int j = 0; j < 3; j++) fb.write(blank.data(), Phonemes::kCount);
  fb.write(ah.data(), Phonemes::kCount);
  for (int j = 0; j < 2; j++) fb.write(blank.data(), Phonemes::kCount);
  fb.write(ah.data(), Phonemes::kCount);
  for (int j = 0; j < 2; j++) fb.write(blank.data(), Phonemes::kCount);
  fb.endOfUtterance();
  std::vector<PhonemeFrame> got;
  PhonemeFrame f;
  while (q.read(f)) got.push_back(f);
  CHECK_EQ(got.size(), 6u);
  if (got.size() == 6) {
    CHECK_EQ(got[0].type, PhonemeFrame::BlankRun);
    CHECK_EQ(got[0].index, 0u);
    CHECK_EQ(got[0].run, 3);
    CHECK_EQ(got[1].type, PhonemeFrame::Frame);
    CHECK_EQ(got[1].index, 3u);
    CHECK_EQ(got[2].run, 2);
    CHECK_EQ(got[2].index, 4u);
    CHECK_EQ(got[3].index, 6u);
    CHECK_EQ(got[4].type, PhonemeFrame::BlankRun);
    CHECK_EQ(got[4].run, 2);
    CHECK_EQ(got[5].type, PhonemeFrame::EndOfUtterance);
    CHECK_EQ(got[5].index, 9u);
  }
  // total frames represented == frames written
  uint32_t total = 0;
  for (auto& g : got) total += g.type == PhonemeFrame::EndOfUtterance ? 0 : g.run;
  CHECK_EQ(total, 9u);
}

TEST(frame_builder_threshold_one_disables_skipping) {
  PhonemeQueue q(64);
  PhonemeQueueWriter w(q);
  FrameBuilder fb(w);
  fb.setBlankThreshold(1.0f);
  auto blank = logp({{0, 0.999f}});
  for (int j = 0; j < 5; j++) fb.write(blank.data(), Phonemes::kCount);
  PhonemeFrame f;
  int frames = 0;
  while (q.read(f)) frames += f.type == PhonemeFrame::Frame;
  CHECK_EQ(frames, 5);
}

TEST_MAIN()
