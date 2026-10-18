#pragma once
#include <math.h>
#include <stdint.h>

#include "TinySTT/dsp/DotProduct.h"

// The inference hot loop: -O3 even in -Os (Arduino) builds
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("O3")
#endif

namespace tinystt {

/**
 * @brief Weights of a dense layer y = W x + b, stored either as float32 or as
 * int8 with one float scale per output row (weight-only quantization:
 * W[r][c] ~= q[r][c] * scale[r]). These are the plain, portable reference
 * kernels; optimized (SIMD) variants must match them.
 */
struct DenseWeights {
  int rows = 0;  ///< outputs
  int cols = 0;  ///< inputs
  const float* w = nullptr;      ///< [rows * cols] float32, or
  const int8_t* q = nullptr;     ///< [rows * cols] int8 ...
  const float* scale = nullptr;  ///< ... with [rows] scales
  const float* bias = nullptr;   ///< [rows], optional
  /// weights are in fast internal RAM (set by AcousticModel's placement):
  /// lets the DSP kernel read two rows per pass, which is slower when the
  /// weights stream from flash or PSRAM (shared QSPI bus)
  bool internal_ram = false;

  bool valid() const { return rows > 0 && cols > 0 && (w != nullptr || (q != nullptr && scale != nullptr)); }
  bool isQuantized() const { return w == nullptr && q != nullptr; }
  uint32_t params() const { return (uint32_t)rows * cols + (bias ? rows : 0); }
  uint32_t macs() const { return (uint32_t)rows * cols; }

  /**
   * int8 weights x int8 activations: x is quantized per call (symmetric,
   * scale = max|x| / 127, round half to even) into `xq` (cols bytes, 16 byte
   * aligned), then int32 dot products (SIMD on ESP32-S3):
   *   y[r] = dot(q[r], xq) * scale[r] * x_scale + b[r]
   * Requires quantized weights; research/model_numpy.py implements the same.
   */
  void applyQ8(const float* x, float* y, int8_t* xq, uint32_t* xprep = nullptr) const {
    float xs = quantizeInput(x, xq);
#if defined(TINYSTT_HAS_DSP_S8)
    if (xprep != nullptr && cols % 4 == 0) {
      applyQ8Dsp(xq, xprep, &xs, y, 1);
      return;
    }
#else
    (void)xprep;
#endif
    for (int r = 0; r < rows; r++) {
      int32_t acc = dotS8(q + (size_t)r * cols, xq, cols);
      y[r] = (float)acc * scale[r] * xs + (bias ? bias[r] : 0.0f);
    }
  }

#if defined(TINYSTT_HAS_DSP_S8)
  /// DSP path: prepares the n quantized inputs (xq [n][cols]) once; weights
  /// in internal RAM are processed two rows per pass, others one row at a
  /// time (sequential reads); xprep needs n * cols / 2 words
  void applyQ8Dsp(const int8_t* xq, uint32_t* xprep, const float* xs, float* y, int n) const {
    const int words = cols / 2;
    for (int b = 0; b < n; b++) prepareS8Dsp(xq + (size_t)b * cols, xprep + (size_t)b * words, cols);
    int r = 0;
    for (; internal_ram && r + 2 <= rows; r += 2) {
      const int8_t* w0 = q + (size_t)r * cols;
      const int8_t* w1 = w0 + cols;
      float s0 = scale[r], s1 = scale[r + 1];
      float b0 = bias ? bias[r] : 0.0f, b1 = bias ? bias[r + 1] : 0.0f;
      for (int b = 0; b < n; b++) {
        int32_t a0, a1;
        dot2S8DspPre(w0, w1, xprep + (size_t)b * words, cols, a0, a1);
        y[(size_t)b * rows + r] = (float)a0 * s0 * xs[b] + b0;
        y[(size_t)b * rows + r + 1] = (float)a1 * s1 * xs[b] + b1;
      }
    }
    for (; r < rows; r++) {
      const int8_t* w0 = q + (size_t)r * cols;
      for (int b = 0; b < n; b++)
        y[(size_t)b * rows + r] = (float)dotS8DspPre(w0, xprep + (size_t)b * words, cols) * scale[r] * xs[b] +
                                  (bias ? bias[r] : 0.0f);
    }
  }
#endif

  /// Quantizes x (cols values) into xq like applyQ8; returns the scale
  float quantizeInput(const float* x, int8_t* xq) const {
    float mx = 0.0f;
    for (int c = 0; c < cols; c++) {
      float a = fabsf(x[c]);
      if (a > mx) mx = a;
    }
    float xs = mx > 0.0f ? mx / 127.0f : 1.0f;
    float inv = 1.0f / xs;
    for (int c = 0; c < cols; c++) {
      float v = rintf(x[c] * inv);
      xq[c] = (int8_t)(v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v));
    }
    return xs;
  }

  /**
   * Batched y[b] = W x[b] (+ bias) for b < n with x [n][cols], y [n][rows].
   * Each weight row is loaded once and used for all n inputs (it stays in
   * the data cache), which divides the memory traffic by n. Results are
   * identical to n separate apply()/applyQ8() calls. With `xq` (n * cols
   * bytes, 16 byte aligned, cols a multiple of 16 for SIMD) the int8
   * activation path is used and `xs` receives the n scales.
   */
  void applyBatch(const float* x, float* y, int n, int8_t* xq = nullptr, float* xs = nullptr,
                  uint32_t* xprep = nullptr) const {
    if (xq != nullptr) {
      for (int b = 0; b < n; b++) xs[b] = quantizeInput(x + (size_t)b * cols, xq + (size_t)b * cols);
#if defined(TINYSTT_HAS_DSP_S8)
      if (xprep != nullptr && cols % 4 == 0) {
        applyQ8Dsp(xq, xprep, xs, y, n);
        return;
      }
#else
      (void)xprep;
#endif
      for (int r = 0; r < rows; r++) {
        const int8_t* row = q + (size_t)r * cols;
        float rb = bias ? bias[r] : 0.0f;
        for (int b = 0; b < n; b++)
          y[(size_t)b * rows + r] = (float)dotS8(row, xq + (size_t)b * cols, cols) * scale[r] * xs[b] + rb;
      }
      return;
    }
    for (int r = 0; r < rows; r++) {
      float rb = bias ? bias[r] : 0.0f;
      for (int b = 0; b < n; b++) {
        const float* xb = x + (size_t)b * cols;
        float acc = 0.0f;
        if (isQuantized()) {
          const int8_t* row = q + (size_t)r * cols;
          for (int c = 0; c < cols; c++) acc += (float)row[c] * xb[c];
          acc *= scale[r];
        } else {
          const float* row = w + (size_t)r * cols;
          for (int c = 0; c < cols; c++) acc += row[c] * xb[c];
        }
        y[(size_t)b * rows + r] = acc + rb;
      }
    }
  }

  /// y = W x (+ b); y may not alias x
  void apply(const float* x, float* y) const {
    if (isQuantized()) {
      for (int r = 0; r < rows; r++) {
        const int8_t* row = q + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += (float)row[c] * x[c];
        y[r] = acc * scale[r] + (bias ? bias[r] : 0.0f);
      }
    } else {
      for (int r = 0; r < rows; r++) {
        const float* row = w + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc + (bias ? bias[r] : 0.0f);
      }
    }
  }
};

inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

inline void reluInPlace(float* x, int n) {
  for (int j = 0; j < n; j++)
    if (x[j] < 0.0f) x[j] = 0.0f;
}

/// In-place log-softmax (numerically stable)
inline void logSoftmaxInPlace(float* x, int n) {
  float mx = x[0];
  for (int j = 1; j < n; j++)
    if (x[j] > mx) mx = x[j];
  float sum = 0.0f;
  for (int j = 0; j < n; j++) sum += expf(x[j] - mx);
  float lse = mx + logf(sum);
  for (int j = 0; j < n; j++) x[j] -= lse;
}

/**
 * @brief Recurrent half of a GRU step: `gi` holds the precomputed input
 * projection W_i x + b_i [3H] (e.g. from a batched applyBatch()); computes
 * W_h h + b_h into `gh` [3H] and updates h in place.
 */
inline void gruRecurrent(const DenseWeights& hh, const float* gi, float* h, int hidden, float* gh,
                         int8_t* xq = nullptr, uint32_t* xprep = nullptr) {
  if (xq != nullptr)
    hh.applyQ8(h, gh, xq, xprep);
  else
    hh.apply(h, gh);
  for (int j = 0; j < hidden; j++) {
    float r = sigmoidf(gi[j] + gh[j]);
    float z = sigmoidf(gi[hidden + j] + gh[hidden + j]);
    float n = tanhf(gi[2 * hidden + j] + r * gh[2 * hidden + j]);
    h[j] = (1.0f - z) * n + z * h[j];
  }
}

/**
 * @brief One GRU step with PyTorch semantics (gate order r, z, n):
 *   r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
 *   z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
 *   n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
 *   h = (1 - z) * n + z * h
 * `scratch` needs 6 * hidden floats; with `xq` (16 byte aligned, >= max
 * input width) both matrix products use int8 activations (applyQ8).
 */
inline void gruStep(const DenseWeights& ih, const DenseWeights& hh, const float* x, float* h,
                    int hidden, float* scratch, int8_t* xq = nullptr) {
  float* gi = scratch;               // [3H] W_i x + b_i
  float* gh = scratch + 3 * hidden;  // [3H] W_h h + b_h
  if (xq != nullptr)  // int8 activations
    ih.applyQ8(x, gi, xq);
  else
    ih.apply(x, gi);
  gruRecurrent(hh, gi, h, hidden, gh, xq);
}

}  // namespace tinystt

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
