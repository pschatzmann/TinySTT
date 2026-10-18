// Compute / memory budget from docs/plan.md, checked against the production
// model shape (AcousticModelConfig defaults) and any model container.
#include "AudioTools.h"
#include "TestData.h"
#include "TestUtil.h"
#include "TinySTT/AcousticModel.h"
#include "TinySTT/PhonemeFrame.h"

using namespace tinystt;

static constexpr uint32_t kMaxParams = 1500000;
static constexpr uint32_t kMaxMacsPerFrame = 1500000;
static constexpr uint32_t kMaxInt8WeightBytes = 1500000;
static constexpr uint32_t kMaxActivationBytes = 64 * 1024;
static constexpr uint32_t kMaxQueueBytes = 1024;

static void checkBudget(const AcousticModelConfig& cfg) {
  std::printf("  params %u, MAC/frame %u (%.1f M MAC/s), activations %u B (batch 4: %u B)\n", cfg.params(),
              cfg.macsPerFrame(), cfg.macsPerFrame() * cfg.outputFrameRate() / 1e6, cfg.activationBytes(),
              cfg.activationBytes(4));
  CHECK(cfg.params() <= kMaxParams);
  CHECK(cfg.macsPerFrame() <= kMaxMacsPerFrame);
  CHECK(cfg.params() <= kMaxInt8WeightBytes);  // int8: one byte per weight (+ scales)
  CHECK(cfg.activationBytes() <= kMaxActivationBytes);
  CHECK(cfg.activationBytes(4) <= kMaxActivationBytes);  // batch of 4 (ESP32 default)
  CHECK(cfg.conv_channels % 16 == 0 && cfg.hidden % 16 == 0);  // SIMD friendly
}

TEST(budget_production_config) { checkBudget(AcousticModelConfig()); }

TEST(budget_queue) {
  CHECK(64 * sizeof(PhonemeFrame) <= kMaxQueueBytes);
}

TEST(budget_counts_match_container) {
  // params() derived from the shape must equal what the container stores
  auto img = readTestFile("placeholder_model.tstt");
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), img.size()), ModelContainer::OK);
  AcousticModelConfig cfg;
  CHECK(cfg.fromContainer(mc));
  uint32_t stored = 0;
  for (int j = 0; j < mc.sectionCount(); j++) {
    auto s = mc.section(j);
    std::string name = s.name;
    bool is_weight_f32 = s.dtype == ModelContainer::F32 && name.find(".scale") == std::string::npos &&
                         name.rfind("features.", 0) != 0;
    if (is_weight_f32) stored += s.count;
  }
  CHECK_EQ(stored, cfg.params());
  checkBudget(cfg);
}

TEST_MAIN()
