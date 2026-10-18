/**
 * On-device self-test: runs the placeholder model (test/data) in every
 * precision, batch size and weight placement and compares the results with
 * the numpy reference (research/model_numpy.py), plus the int8 dot product
 * kernels against the portable reference. Prints PASS/FAIL per check.
 * Regenerate the data with: python research/make_selftest_data.py
 */
#include "AudioTools.h"
#include "TinySTT.h"
#include "selftest_data.h"

using P = AcousticModel::Precision;
using L = AcousticModel::Placement;

ModelContainer model;
int failures = 0;
uint32_t T, n_mels, out_frames, classes;
const float *feats, *lp32, *lp8, *lp8a;

void check(bool ok, const char* what) {
  Serial.printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) failures++;
}

float runAndCompare(P precision, L placement, int batch, const float* golden) {
  AcousticModel m;
  m.setBatchFrames(batch);
  if (!m.begin(model, precision, placement)) return 1e9f;
  float max_diff = 0;
  uint32_t frame = 0;
  auto take = [&](int n) {
    for (int i = 0; i < n; i++, frame++)
      for (uint32_t c = 0; c < classes; c++)
        max_diff = fmaxf(max_diff, fabsf(m.logProbs(i)[c] - golden[frame * classes + c]));
  };
  for (uint32_t t = 0; t < T; t++) take(m.write(feats + t * n_mels));
  take(m.flush());
  return frame == out_frames ? max_diff : 1e9f;
}

void runSelfTest() {
  Serial.printf("TinySTT self-test, int8 kernel: %s\n", simdS8Name());

  // dot product kernels
  alignas(16) static int8_t a[512], b[512];
  int bad = 0;
  for (int t = 0; t < 300; t++) {
    int len = 16 * (1 + random(0, 32));
    for (int i = 0; i < len; i++) { a[i] = (int8_t)random(-128, 128); b[i] = (int8_t)random(-128, 128); }
    if (dotS8(a, b, len) != dotS8Reference(a, b, len)) bad++;
  }
  check(bad == 0, "dotS8 == reference (300 random vectors)");

  check(model.begin(selftest_model, selftest_model_len) == ModelContainer::OK, "model container");
  const uint32_t* hdr = (const uint32_t*)selftest_golden;
  T = hdr[0]; n_mels = hdr[1]; out_frames = hdr[2]; classes = hdr[3];
  feats = (const float*)(selftest_golden + 16);
  lp32 = feats + T * n_mels;
  lp8 = lp32 + out_frames * classes;
  lp8a = lp8 + out_frames * classes;

  struct Case { P p; const float* golden; const char* name; } cases[] = {
      {P::Float32, lp32, "float32"}, {P::Int8, lp8, "int8 weights"}, {P::Int8Activations, lp8a, "int8 activations"}};
  for (auto& c : cases) {
    for (L placement : {L::InPlace, L::ExternalRAM, L::Auto}) {
      for (int batch : {1, 4}) {
        float d = runAndCompare(c.p, placement, batch, c.golden);
        char what[96];
        snprintf(what, sizeof(what), "%-16s %-7s batch %d: max diff %.2e", c.name,
                 placement == L::InPlace ? "inplace" : placement == L::ExternalRAM ? "psram" : "auto", batch, d);
        check(d < 1e-3f, what);
      }
    }
  }
  Serial.printf("--- self-test %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
}

void setup() { Serial.begin(115200); }

void loop() {
  // runs once as soon as the serial port is open (USB serial on RP2040/RP2350
  // loses output written before the host connects; UARTs start at once)
  static bool done = false;
  if (!done && Serial) {
    delay(1500);
    runSelfTest();
    done = true;
  }
  delay(100);
}
