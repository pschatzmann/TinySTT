#pragma once
#include <stdlib.h>
#include <string.h>

#include <vector>

#if defined(ESP32)
#include "esp_heap_caps.h"
#endif

#include "AudioTools.h"
#include "TinySTT/ModelContainer.h"
#include "TinySTT/Ops.h"
#include "TinySTT/Phonemes.h"

namespace tinystt {

/**
 * @brief Shape of the acoustic model. Stored as the "model.config" U32 section
 * of the model container: [version, n_mels, conv_channels, conv_kernel,
 * conv_stride, gru_layers, hidden, n_classes]. The defaults are the target
 * production size checked against the budget in docs/plan.md.
 */
struct AcousticModelConfig {
  static constexpr uint32_t kVersion = 1;
  uint32_t n_mels = 80;
  uint32_t conv_channels = 256;
  uint32_t conv_kernel = 5;
  uint32_t conv_stride = 3;  ///< 10 ms features -> 30 ms output frames
  uint32_t gru_layers = 2;
  uint32_t hidden = 256;
  uint32_t n_classes = Phonemes::kCount;

  bool fromContainer(const ModelContainer& mc) {
    auto s = mc.find("model.config");
    if (!s.valid() || s.u32() == nullptr || s.count < 8) return false;
    const uint32_t* v = s.u32();
    if (v[0] != kVersion) return false;
    n_mels = v[1];
    conv_channels = v[2];
    conv_kernel = v[3];
    conv_stride = v[4];
    gru_layers = v[5];
    hidden = v[6];
    n_classes = v[7];
    return true;
  }

  /// Weights + biases
  uint32_t params() const {
    uint32_t p = conv_channels * n_mels * conv_kernel + conv_channels;
    for (uint32_t l = 0; l < gru_layers; l++) {
      uint32_t in = l == 0 ? conv_channels : hidden;
      p += 3 * hidden * (in + hidden) + 6 * hidden;
    }
    p += n_classes * hidden + n_classes;
    return p;
  }

  /// Multiply-accumulates per output (30 ms) frame
  uint32_t macsPerFrame() const {
    uint32_t m = conv_channels * n_mels * conv_kernel;
    for (uint32_t l = 0; l < gru_layers; l++) {
      uint32_t in = l == 0 ? conv_channels : hidden;
      m += 3 * hidden * (in + hidden);
    }
    m += n_classes * hidden;
    return m;
  }

  /// Runtime state + scratch in bytes for `batch` frames per forward pass
  uint32_t activationBytes(uint32_t batch = 1) const {
    uint32_t floats = conv_kernel * n_mels                        // input window
                      + batch * n_mels * conv_kernel              // flattened windows
                      + batch * conv_channels                     // conv outputs
                      + gru_layers * hidden + batch * hidden      // hidden states, layer outputs
                      + batch * 3 * hidden + 3 * hidden           // input projections, recurrent scratch
                      + batch * n_classes;                        // logits
    uint32_t max_in = n_mels * conv_kernel > hidden ? n_mels * conv_kernel : hidden;
    return floats * 4 + batch * max_in;                            // + int8 activations
  }

  /// Output frames per second for 100 feature frames per second
  float outputFrameRate() const { return 100.0f / conv_stride; }
};

/**
 * @brief Streaming causal acoustic model: features (80 log-mel per 10 ms) ->
 * CTC log-probabilities over Phonemes (one output per conv_stride inputs).
 *
 *   conv1d(n_mels -> C, kernel K, stride S, causal: K-1 left zero padding) + ReLU
 *   -> L x GRU(H) -> linear(H -> n_classes) -> log-softmax
 *
 * Weights are read in place from a ModelContainer, as float32 ("<name>") or
 * int8 + per-row scale ("<name>.q", "<name>.scale"). Section names follow the
 * PyTorch state_dict: conv.weight [C, n_mels, K], conv.bias, gru{l}.weight_ih,
 * gru{l}.weight_hh, gru{l}.bias_ih, gru{l}.bias_hh, out.weight, out.bias.
 */
class AcousticModel {
 public:
  /**
   * Float32: float weights and activations.
   * Int8: int8 weights, float activations (reference for the weight-only QAT).
   * Int8Activations: int8 weights and per-vector quantized int8 activations
   *   with int32 dot products (SIMD on ESP32-S3) - the fast path.
   * Auto: Int8Activations if the container has int8 weights, else Float32.
   */
  enum class Precision { Auto, Float32, Int8, Int8Activations };

  /**
   * InPlace: weights are used where they are (container in flash/RAM).
   * ExternalRAM: weights are copied to PSRAM (ESP32, RP2350 with PSRAM).
   * Auto: on microcontrollers, promote the GRU recurrent weights to internal
   *   RAM while keeping internalRamReserve() free, and copy the rest to PSRAM
   *   if there is some; InPlace on the desktop. Bandwidth, not arithmetic, limits the int8 path:
   *   ESP32-S3 measured ~27 MB/s flash, ~59 MB/s PSRAM, >600 MB/s internal.
   */
  enum class Placement { Auto, InPlace, ExternalRAM };

  AcousticModel() = default;
  AcousticModel(const AcousticModel&) = delete;
  AcousticModel& operator=(const AcousticModel&) = delete;
  ~AcousticModel() { freeCopies(); }

  /**
   * Output frames computed together (default 1). With n > 1 the conv, input
   * projection and output weights are read once per n frames, which cuts the
   * memory traffic (the bottleneck on microcontrollers) at the cost of
   * (n - 1) x 30 ms extra latency. Results are identical. Call before begin().
   */
  void setBatchFrames(int n) { batch = n < 1 ? 1 : n; }
  int batchFrames() const { return batch; }

  /// Internal RAM that Placement::Auto leaves free (default 96 KB)
  void setInternalRamReserve(size_t bytes) { internal_reserve = bytes; }
  size_t internalRamReserve() const { return internal_reserve; }

  /// Binds the weights; the container must outlive the model.
  bool begin(const ModelContainer& mc, Precision precision = Precision::Auto,
             Placement placement = Placement::Auto) {
    is_ready = false;
    freeCopies();
    if (!mc || !cfg.fromContainer(mc)) {
      LOGE("TinySTT: model.config missing or unsupported");
      return false;
    }
    bool ok = true;
    if (precision == Precision::Auto)
      precision = mc.has("conv.weight.q") ? Precision::Int8Activations : Precision::Float32;
    act_quant = precision == Precision::Int8Activations;
    Precision weights = act_quant ? Precision::Int8 : precision;
    ok &= bind(mc, "conv.weight", "conv.bias", cfg.conv_channels, cfg.n_mels * cfg.conv_kernel, weights, conv);
    gru_ih.resize(cfg.gru_layers);
    gru_hh.resize(cfg.gru_layers);
    for (uint32_t l = 0; l < cfg.gru_layers && ok; l++) {
      char w[32], b[32];
      uint32_t in = l == 0 ? cfg.conv_channels : cfg.hidden;
      snprintf(w, sizeof(w), "gru%u.weight_ih", (unsigned)l);
      snprintf(b, sizeof(b), "gru%u.bias_ih", (unsigned)l);
      ok &= bind(mc, w, b, 3 * cfg.hidden, in, weights, gru_ih[l]);
      snprintf(w, sizeof(w), "gru%u.weight_hh", (unsigned)l);
      snprintf(b, sizeof(b), "gru%u.bias_hh", (unsigned)l);
      ok &= bind(mc, w, b, 3 * cfg.hidden, cfg.hidden, weights, gru_hh[l]);
    }
    ok &= bind(mc, "out.weight", "out.bias", cfg.n_classes, cfg.hidden, weights, out);
    if (!ok) return false;
    if (act_quant) {
      // SIMD needs 16 byte aligned rows: widths must be multiples of 16
      uint32_t widths[] = {cfg.n_mels * cfg.conv_kernel, cfg.conv_channels, cfg.hidden};
      for (uint32_t w : widths)
        if (w % kSimdAlign != 0) LOGW("TinySTT: width %u not a multiple of 16 - no SIMD", (unsigned)w);
      size_t n = cfg.n_mels * cfg.conv_kernel;
      if (cfg.conv_channels > n) n = cfg.conv_channels;
      if (cfg.hidden > n) n = cfg.hidden;
      xq_buf.assign((size_t)batch * n + kSimdAlign, 0);
      xq = (int8_t*)(((uintptr_t)xq_buf.data() + kSimdAlign - 1) & ~(uintptr_t)(kSimdAlign - 1));
#if defined(TINYSTT_HAS_DSP_S8)
      xprep_buf.assign((size_t)batch * (n / 2 + 2), 0);  // prepared 16 bit pairs (DSP kernel)
#endif
    } else {
      xq = nullptr;
    }
    window.assign(cfg.conv_kernel * cfg.n_mels, 0.0f);
    flat.assign((size_t)batch * cfg.n_mels * cfg.conv_kernel, 0.0f);
    conv_out.assign((size_t)batch * cfg.conv_channels, 0.0f);
    hidden_state.assign(cfg.gru_layers * cfg.hidden, 0.0f);
    layer_out.assign((size_t)batch * cfg.hidden, 0.0f);
    gi.assign((size_t)batch * 3 * cfg.hidden, 0.0f);
    scratch.assign(3 * cfg.hidden, 0.0f);
    logits.assign((size_t)batch * cfg.n_classes, 0.0f);
    xs.assign(batch, 1.0f);
    // last: the internal RAM reserve must see all of the model's own buffers
    place(placement);
    reset();
    is_ready = true;
    return true;
  }

  /// Clears the recurrent state and the input window (new utterance)
  void reset() {
    std::fill(window.begin(), window.end(), 0.0f);
    std::fill(hidden_state.begin(), hidden_state.end(), 0.0f);
    input_frames = 0;
    output_frames = 0;
    pending = 0;
  }

  /**
   * Adds one feature frame (n_mels values). Every conv_stride inputs
   * (starting with the first) an output frame is due; they are computed in
   * batches of batchFrames(). Returns the number of new output frames
   * (0 or batchFrames()); logProbs(i) holds their log-probabilities.
   */
  int write(const float* features) {
    if (!is_ready) return 0;
    // shift the window by one frame (oldest first) and append
    int n = cfg.n_mels, k = cfg.conv_kernel;
    memmove(window.data(), window.data() + n, (size_t)(k - 1) * n * sizeof(float));
    memcpy(window.data() + (size_t)(k - 1) * n, features, n * sizeof(float));
    bool emit = (input_frames % cfg.conv_stride) == 0;
    input_frames++;
    if (!emit) return 0;
    // PyTorch Conv1d weight [C][n_mels][K] flattened: index i * K + kk
    float* f = flat.data() + (size_t)pending * n * k;
    for (int i = 0; i < n; i++)
      for (int kk = 0; kk < k; kk++) f[i * k + kk] = window[kk * n + i];
    if (++pending < batch) return 0;
    return flush();
  }

  /// Computes pending output frames of an incomplete batch (end of
  /// utterance); returns their number
  int flush() {
    if (!is_ready || pending == 0) return 0;
    int n = pending;
    forward(n);
    pending = 0;
    output_frames += n;
    return n;
  }

  /// Bytes of weights copied to internal RAM / PSRAM by the placement
  size_t internalWeightBytes() const { return internal_bytes; }
  size_t psramWeightBytes() const { return psram_bytes; }
  bool usesInt8Activations() const { return act_quant; }

  /// Log-probabilities of output frame i of the last write()/flush() result
  const float* logProbs(int i = 0) const { return logits.data() + (size_t)i * cfg.n_classes; }
  int classCount() const { return cfg.n_classes; }
  const AcousticModelConfig& config() const { return cfg; }
  bool isQuantized() const { return conv.isQuantized(); }
  uint32_t inputFrames() const { return input_frames; }
  uint32_t outputFrames() const { return output_frames; }
  operator bool() const { return is_ready; }

 protected:
  AcousticModelConfig cfg;
  DenseWeights conv, out;
  std::vector<DenseWeights> gru_ih, gru_hh;
  std::vector<float> window, flat, conv_out, hidden_state, layer_out, gi, scratch, logits, xs;
  std::vector<int8_t> xq_buf;
  std::vector<uint32_t> xprep_buf;
  int batch = 1;
  int pending = 0;
  int8_t* xq = nullptr;  ///< 16 byte aligned view into xq_buf
  bool act_quant = false;
  size_t internal_reserve = 96 * 1024;
  size_t internal_bytes = 0, psram_bytes = 0;
  std::vector<void*> copies;
  uint32_t input_frames = 0;
  uint32_t output_frames = 0;
  bool is_ready = false;

#ifdef TINYSTT_PROFILE
 public:
  /// accumulated microseconds per stage (TINYSTT_PROFILE builds only)
  uint32_t prof_conv_us = 0, prof_gru_us = 0, prof_out_us = 0;
  void resetProfile() { prof_conv_us = prof_gru_us = prof_out_us = 0; }

 protected:
#define TINYSTT_PROF_START() uint32_t _prof_t = micros()
#define TINYSTT_PROF_LAP(var)        \
  do {                               \
    uint32_t _now = micros();        \
    var += _now - _prof_t;           \
    _prof_t = _now;                  \
  } while (0)
#else
#define TINYSTT_PROF_START()
#define TINYSTT_PROF_LAP(var)
#endif

  void forward(int n) {
    TINYSTT_PROF_START();
    int8_t* q = act_quant ? xq : nullptr;
    uint32_t* xp = act_quant && !xprep_buf.empty() ? xprep_buf.data() : nullptr;
    conv.applyBatch(flat.data(), conv_out.data(), n, q, xs.data(), xp);
    reluInPlace(conv_out.data(), n * cfg.conv_channels);
    TINYSTT_PROF_LAP(prof_conv_us);
    const float* x = conv_out.data();
    for (uint32_t l = 0; l < cfg.gru_layers; l++) {
      gru_ih[l].applyBatch(x, gi.data(), n, q, xs.data(), xp);  // input projections of all n frames
      float* h = hidden_state.data() + l * cfg.hidden;
      for (int b = 0; b < n; b++) {  // the recurrence is sequential
        gruRecurrent(gru_hh[l], gi.data() + (size_t)b * 3 * cfg.hidden, h, cfg.hidden, scratch.data(), q, xp);
        memcpy(layer_out.data() + (size_t)b * cfg.hidden, h, cfg.hidden * sizeof(float));
      }
      x = layer_out.data();
    }
    TINYSTT_PROF_LAP(prof_gru_us);
    out.applyBatch(x, logits.data(), n, q, xs.data(), xp);
    for (int b = 0; b < n; b++) logSoftmaxInPlace(logits.data() + (size_t)b * cfg.n_classes, cfg.n_classes);
    TINYSTT_PROF_LAP(prof_out_us);
  }

  // ---- weight placement (platform specific memory) ----

  /// true if external RAM (PSRAM) can be allocated
  static bool hasExternalRam() {
#if defined(ESP32)
    return heap_caps_get_free_size(MALLOC_CAP_SPIRAM) > 0;
#elif defined(ARDUINO_ARCH_RP2040) && defined(RP2350_PSRAM_CS)
    return rp2040.getPSRAMSize() > 0;
#else
    return false;
#endif
  }

  /// free internal RAM and the largest block that can be allocated there
  static void internalRam(size_t& free_bytes, size_t& largest) {
#if defined(ESP32)
    const uint32_t caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    free_bytes = heap_caps_get_free_size(caps);
    largest = heap_caps_get_largest_free_block(caps);
#elif defined(ARDUINO_ARCH_RP2040)
    free_bytes = largest = (size_t)rp2040.getFreeHeap();
#else
    free_bytes = largest = 0;  // desktop: weights stay where they are
#endif
  }

  /// raw allocation in internal RAM or external RAM (nullptr if unavailable)
  static void* allocRaw(size_t n, bool internal) {
#if defined(ESP32)
    return heap_caps_malloc(n, internal ? (MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : MALLOC_CAP_SPIRAM);
#elif defined(ARDUINO_ARCH_RP2040)
#if defined(RP2350_PSRAM_CS)
    if (!internal) return pmalloc(n);
#endif
    return internal ? malloc(n) : nullptr;
#else
    (void)internal;
    return malloc(n);
#endif
  }

  static void freeRaw(void* p) {
#if defined(ESP32)
    heap_caps_free(p);
#else
    free(p);  // also releases pmalloc() memory on the RP2350
#endif
  }

  void freeCopies() {
    for (void* p : copies) freeRaw(p);
    copies.clear();
    internal_bytes = psram_bytes = 0;
  }

  static size_t weightBytes(const DenseWeights& d) {
    return (size_t)d.rows * d.cols * (d.isQuantized() ? 1 : sizeof(float));
  }

  /// Copies the weight matrix of `d` to internal RAM or external RAM
  /// (16 byte aligned for the SIMD kernels)
  bool copyWeights(DenseWeights& d, bool internal) {
    size_t n = weightBytes(d);
    void* raw = allocRaw(n + kSimdAlign, internal);
    if (raw == nullptr) return false;
    void* p = (void*)(((uintptr_t)raw + kSimdAlign - 1) & ~(uintptr_t)(kSimdAlign - 1));
    if (d.isQuantized()) {
      memcpy(p, d.q, n);
      d.q = (const int8_t*)p;
    } else {
      memcpy(p, d.w, n);
      d.w = (const float*)p;
    }
    copies.push_back(raw);
    d.internal_ram = internal;
    (internal ? internal_bytes : psram_bytes) += n;
    return true;
  }

  void place(Placement placement) {
#if defined(ESP32) || defined(ARDUINO_ARCH_RP2040)
    if (placement == Placement::InPlace) return;
    std::vector<DenseWeights*> rest = {&conv, &out};
    for (auto& d : gru_ih) rest.push_back(&d);
    if (placement == Placement::Auto) {
      // recurrent weights first: read every frame in any schedule
      for (auto& d : gru_hh) {
        size_t n = weightBytes(d), free_bytes = 0, largest = 0;
        internalRam(free_bytes, largest);
        bool fits = largest >= n + 2 * kSimdAlign && free_bytes >= n + internal_reserve;
        if (!(fits && copyWeights(d, true))) rest.push_back(&d);
      }
    } else {
      for (auto& d : gru_hh) rest.push_back(&d);
    }
    if (hasExternalRam()) {
      for (DenseWeights* d : rest)
        if (!copyWeights(*d, false)) LOGW("TinySTT: external RAM copy failed - weights stay in place");
    }
    LOGI("TinySTT: weights %u B internal, %u B external RAM", (unsigned)internal_bytes, (unsigned)psram_bytes);
#else
    (void)placement;
#endif
  }

  static bool bind(const ModelContainer& mc, const char* weight, const char* bias, uint32_t rows,
                   uint32_t cols, Precision precision, DenseWeights& d) {
    d = DenseWeights();
    d.rows = rows;
    d.cols = cols;
    char qname[48], sname[48];
    snprintf(qname, sizeof(qname), "%s.q", weight);
    snprintf(sname, sizeof(sname), "%s.scale", weight);
    auto wf = mc.find(weight);
    auto wq = mc.find(qname);
    auto ws = mc.find(sname);
    bool have_f32 = wf.valid() && wf.f32() && wf.count == rows * cols;
    bool have_q8 = wq.valid() && wq.i8() && wq.count == rows * cols && ws.valid() && ws.f32() &&
                   ws.count == rows;
    bool use_q8 = precision == Precision::Int8 || (precision == Precision::Auto && have_q8);
    if (use_q8 && have_q8) {
      d.q = wq.i8();
      d.scale = ws.f32();
    } else if (!use_q8 && have_f32) {
      d.w = wf.f32();
    } else {
      LOGE("TinySTT: weights '%s' missing or wrong shape (%u x %u)", weight, (unsigned)rows, (unsigned)cols);
      return false;
    }
    auto b = mc.find(bias);
    if (b.valid()) {
      if (!b.f32() || b.count != rows) {
        LOGE("TinySTT: bias '%s' has wrong shape", bias);
        return false;
      }
      d.bias = b.f32();
    }
    return true;
  }
};

}  // namespace tinystt
