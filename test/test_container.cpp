// ModelContainer: writer/reader round trip, CRC, version and corruption checks.
#include "AudioTools.h"
#include "TestUtil.h"
#include "TinySTT/ModelContainer.h"

using namespace tinystt;

static std::vector<uint8_t> sample() {
  ModelContainerWriter w;
  float f[5] = {1.5f, -2.0f, 0.0f, 3.25f, 1e-6f};
  int8_t q[3] = {-128, 0, 127};
  uint32_t u[2] = {7, 0xFFFFFFFF};
  w.addF32("conv0.weight", f, 5);
  w.addI8("conv0.weight.q", q, 3);
  w.addU32("meta.dims", u, 2);
  w.addString("info", "tinystt test");
  return w.build();
}

TEST(container_round_trip) {
  auto img = sample();
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), img.size()), ModelContainer::OK);
  CHECK(mc);
  CHECK_EQ(mc.sectionCount(), 4);
  CHECK_EQ(mc.version(), ModelContainer::kVersion);
  auto s = mc.find("conv0.weight");
  CHECK(s.valid());
  CHECK_EQ(s.count, 5u);
  CHECK(s.f32() != nullptr);
  CHECK_NEAR(s.f32()[3], 3.25, 0);
  CHECK(s.i8() == nullptr);  // dtype checked
  auto q = mc.find("conv0.weight.q");
  CHECK_EQ(q.i8()[0], -128);
  CHECK_EQ(q.i8()[2], 127);
  CHECK_EQ(mc.find("meta.dims").u32()[1], 0xFFFFFFFFu);
  CHECK_STR((const char*)mc.find("info").data, "tinystt test");
  CHECK(!mc.has("missing"));
}

TEST(container_alignment) {
  auto img = sample();
  ModelContainer mc;
  mc.begin(img.data(), img.size());
  for (int j = 0; j < mc.sectionCount(); j++) {
    CHECK_EQ((size_t)(mc.section(j).data - img.data()) % ModelContainer::kAlign, 0u);
  }
  CHECK_EQ(img.size() % ModelContainer::kAlign, 0u);
}

TEST(container_crc_detects_corruption) {
  auto img = sample();
  img[img.size() - 20] ^= 0x01;
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), img.size()), ModelContainer::BadCrc);
  CHECK(!mc);
  // without CRC verification it still parses (used for trusted mmapped flash)
  CHECK_EQ(mc.begin(img.data(), img.size(), false), ModelContainer::OK);
}

TEST(container_rejects_bad_headers) {
  auto img = sample();
  ModelContainer mc;
  CHECK_EQ(mc.begin(img.data(), 8), ModelContainer::TooSmall);
  auto bad_magic = img;
  bad_magic[0] = 'X';
  CHECK_EQ(mc.begin(bad_magic.data(), bad_magic.size()), ModelContainer::BadMagic);
  auto bad_version = img;
  bad_version[4] = 99;
  CHECK_EQ(mc.begin(bad_version.data(), bad_version.size()), ModelContainer::BadVersion);
  CHECK_EQ(mc.begin(img.data(), img.size() - 16), ModelContainer::BadSize);
  auto bad_section = img;
  // section 0 size field -> beyond end; fix CRC so only the section check fires
  uint8_t* e = &bad_section[ModelContainer::kHeaderSize];
  e[ModelContainer::kNameSize + 8 + 3] = 0x7F;
  uint32_t crc = crc32(&bad_section[16], bad_section.size() - 16);
  memcpy(&bad_section[12], &crc, 4);
  CHECK_EQ(mc.begin(bad_section.data(), bad_section.size()), ModelContainer::BadSection);
}

TEST(container_load_from_stream) {
  auto img = sample();
  MemoryStream in(img.data(), img.size());
  in.begin();
  ModelContainer mc;
  CHECK_EQ(mc.begin(in, img.size()), ModelContainer::OK);
  CHECK_NEAR(mc.find("conv0.weight").f32()[0], 1.5, 0);
  // the loaded copy is independent of the source buffer
  img.assign(img.size(), 0);
  CHECK_NEAR(mc.find("conv0.weight").f32()[0], 1.5, 0);
}

TEST(crc32_matches_zlib) {
  const char* s = "123456789";
  CHECK_EQ(crc32((const uint8_t*)s, 9), 0xCBF43926u);
  // running checksum equals one-shot
  uint32_t c = crc32((const uint8_t*)s, 4);
  CHECK_EQ(crc32((const uint8_t*)s + 4, 5, c), 0xCBF43926u);
  CHECK_EQ(crc8((const uint8_t*)s, 9), 0xF4);
}

TEST_MAIN()
