#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(ESP32)
#include "sdkconfig.h"
#endif

#if defined(CONFIG_IDF_TARGET_ESP32S3)
extern "C" int32_t esp_nn_dot_s8_aligned_esp32s3(const int8_t* a, const int8_t* b, int32_t len);
#define TINYSTT_HAS_SIMD_S8 1
#elif defined(CONFIG_IDF_TARGET_ESP32P4)
#define TINYSTT_HAS_SIMD_S8 1
#define TINYSTT_P4_PIE 1
#endif

namespace tinystt {

/// Alignment and length granularity required by the SIMD int8 kernel
static constexpr int kSimdAlign = 16;

/// Portable reference: sum a[i] * b[i] with int32 accumulation
inline int32_t dotS8Reference(const int8_t* a, const int8_t* b, int len) {
  int32_t acc = 0;
  for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

#if defined(__GNUC__) && !defined(__clang__)
#define TINYSTT_OPTIMIZE_O3 __attribute__((optimize("O3")))
#else
#define TINYSTT_OPTIMIZE_O3
#endif

/**
 * Portable fast path: four independent int32 accumulators, compiled with -O3
 * even in -Os (Arduino) builds. 1.8x faster than the plain loop on an
 * ESP32-P4. Bit identical to dotS8Reference().
 */
TINYSTT_OPTIMIZE_O3 inline int32_t dotS8Unrolled(const int8_t* a, const int8_t* b, int len) {
  int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i = 0;
  for (; i + 4 <= len; i += 4) {
    s0 += (int32_t)a[i] * b[i];
    s1 += (int32_t)a[i + 1] * b[i + 1];
    s2 += (int32_t)a[i + 2] * b[i + 2];
    s3 += (int32_t)a[i + 3] * b[i + 3];
  }
  for (; i < len; i++) s0 += (int32_t)a[i] * b[i];
  return s0 + s1 + s2 + s3;
}

#if defined(TINYSTT_P4_PIE)
// ESP32-P4 PIE (xespv) kernel, the same instruction pattern as Espressif's
// esp-nn esp32p4 fully-connected kernel: 128 bit loads with post increment,
// int8 multiply-accumulate into the 40 bit XACC accumulator, two register
// pairs software pipelined (32 bytes per iteration).
#define TINYSTT_P4_VLD(q, p) asm volatile("esp.vld.128.ip " #q ", %0, 16" : "+r"(p) : : "memory")
#define TINYSTT_P4_VMULAS(qa, qb) asm volatile("esp.vmulas.s8.xacc " #qa ", " #qb : : : "memory")
#define TINYSTT_P4_VMULAS_LD(qn, p, qa, qb) \
  asm volatile("esp.vmulas.s8.xacc.ld.ip " #qn ", %0, 16, " #qa ", " #qb : "+r"(p) : : "memory")

/// Requires a and b 16 byte aligned and len a multiple of 16
inline int32_t dotS8P4(const int8_t* a, const int8_t* b, int len) {
  // enable the PIE unit for this task and select its configuration, exactly
  // like esp-nn does before using it
  asm volatile("csrsi 0x7f2, 1" ::: "memory");
  asm volatile("esp.movx.w.cfg %0" : : "r"(2));
  asm volatile("esp.zero.xacc" ::: "memory");
  int done = 0;
  if (len >= 32) {
    TINYSTT_P4_VLD(q0, a);
    TINYSTT_P4_VLD(q2, a);
    TINYSTT_P4_VLD(q1, b);
    TINYSTT_P4_VLD(q3, b);
    done = 32;
    while (done + 32 <= len) {  // q0..q3 hold the current 32 bytes
      TINYSTT_P4_VMULAS_LD(q0, a, q0, q1);  // acc += q0*q1, q0 = next a
      TINYSTT_P4_VLD(q1, b);
      TINYSTT_P4_VMULAS_LD(q2, a, q2, q3);
      TINYSTT_P4_VLD(q3, b);
      done += 32;
    }
    TINYSTT_P4_VMULAS(q0, q1);
    TINYSTT_P4_VMULAS(q2, q3);
  }
  if (done + 16 <= len) {
    TINYSTT_P4_VLD(q0, a);
    TINYSTT_P4_VLD(q1, b);
    TINYSTT_P4_VMULAS(q0, q1);
  }
  int32_t result;
  asm volatile("esp.movx.r.xacc.l %0" : "=r"(result) : : "memory");
  return result;
}
#endif

#if defined(__ARM_NEON) && !defined(TINYSTT_HAS_SIMD_S8)
#include <arm_neon.h>
#define TINYSTT_HAS_NEON_S8 1
/**
 * ARM NEON (e.g. Raspberry Pi 3/4/5, 32 or 64 bit): 16 bytes per step,
 * vmull_s8 (8 x int16 products) + vpadalq_s16 (pairwise add into int32); with
 * the dot product extension (Pi 5, ARMv8.2+) a single vdotq_s32.
 * Bit identical to dotS8Reference().
 */
TINYSTT_OPTIMIZE_O3 inline int32_t dotS8Neon(const int8_t* a, const int8_t* b, int len) {
  int32x4_t acc = vdupq_n_s32(0);
  int i = 0;
  for (; i + 16 <= len; i += 16) {
    int8x16_t va = vld1q_s8(a + i), vb = vld1q_s8(b + i);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
  }
#if defined(__aarch64__)
  int32_t sum = vaddvq_s32(acc);
#else
  int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  int32_t sum = vget_lane_s32(vpadd_s32(s2, s2), 0);
#endif
  for (; i < len; i++) sum += (int32_t)a[i] * b[i];
  return sum;
}
#endif

#if defined(__ARM_FEATURE_DSP) && defined(__ARM_ARCH) && !defined(__aarch64__) && !defined(TINYSTT_HAS_NEON_S8)
#define TINYSTT_HAS_DSP_S8 1
/**
 * Cortex-M DSP extension (e.g. RP2350 Cortex-M33): 4 bytes per load, unpacked
 * into two signed 16-bit pairs (SXTB16) and accumulated with SMLAD (two
 * 16x16 multiply-accumulates per instruction). No alignment requirement.
 * Bit identical to dotS8Reference().
 */
TINYSTT_OPTIMIZE_O3 inline int32_t dotS8Dsp(const int8_t* a, const int8_t* b, int len) {
  int32_t acc = 0;
  int i = 0;
  for (; i + 4 <= len; i += 4) {
    uint32_t wa, wb, a02, a13, b02, b13;
    memcpy(&wa, a + i, 4);
    memcpy(&wb, b + i, 4);
    asm("sxtb16 %0, %1" : "=r"(a02) : "r"(wa));
    asm("sxtb16 %0, %1, ror #8" : "=r"(a13) : "r"(wa));
    asm("sxtb16 %0, %1" : "=r"(b02) : "r"(wb));
    asm("sxtb16 %0, %1, ror #8" : "=r"(b13) : "r"(wb));
    asm("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(a02), "r"(b02));
    asm("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(a13), "r"(b13));
  }
  for (; i < len; i++) acc += (int32_t)a[i] * b[i];
  return acc;
}

/**
 * Matrix-vector helpers for the DSP extension: the activation vector x (used
 * for every row) is unpacked once into 16-bit pairs, xp[2k] = bytes 0/2 and
 * xp[2k+1] = bytes 1/3 of x[4k..4k+3] (len / 2 words). Then per 4 weight
 * bytes only one load, two SXTB16 and two SMLAD are needed.
 */
inline void prepareS8Dsp(const int8_t* x, uint32_t* xp, int len) {
  for (int i = 0, k = 0; i + 4 <= len; i += 4, k += 2) {
    uint32_t w;
    memcpy(&w, x + i, 4);
    asm("sxtb16 %0, %1" : "=r"(xp[k]) : "r"(w));
    asm("sxtb16 %0, %1, ror #8" : "=r"(xp[k + 1]) : "r"(w));
  }
}

/// Two rows against a prepared vector (len a multiple of 4)
TINYSTT_OPTIMIZE_O3 inline void dot2S8DspPre(const int8_t* w0, const int8_t* w1, const uint32_t* xp, int len,
                                             int32_t& r0, int32_t& r1) {
  int32_t a0 = 0, a1 = 0;
  for (int i = 0, k = 0; i < len; i += 4, k += 2) {
    uint32_t u, v, u02, u13, v02, v13;
    uint32_t x02 = xp[k], x13 = xp[k + 1];
    memcpy(&u, w0 + i, 4);
    memcpy(&v, w1 + i, 4);
    asm("sxtb16 %0, %1" : "=r"(u02) : "r"(u));
    asm("sxtb16 %0, %1, ror #8" : "=r"(u13) : "r"(u));
    asm("sxtb16 %0, %1" : "=r"(v02) : "r"(v));
    asm("sxtb16 %0, %1, ror #8" : "=r"(v13) : "r"(v));
    asm("smlad %0, %1, %2, %0" : "+r"(a0) : "r"(u02), "r"(x02));
    asm("smlad %0, %1, %2, %0" : "+r"(a0) : "r"(u13), "r"(x13));
    asm("smlad %0, %1, %2, %0" : "+r"(a1) : "r"(v02), "r"(x02));
    asm("smlad %0, %1, %2, %0" : "+r"(a1) : "r"(v13), "r"(x13));
  }
  r0 = a0;
  r1 = a1;
}

/// One row against a prepared vector (len a multiple of 4)
TINYSTT_OPTIMIZE_O3 inline int32_t dotS8DspPre(const int8_t* w, const uint32_t* xp, int len) {
  int32_t a = 0;
  for (int i = 0, k = 0; i < len; i += 4, k += 2) {
    uint32_t u, u02, u13;
    memcpy(&u, w + i, 4);
    asm("sxtb16 %0, %1" : "=r"(u02) : "r"(u));
    asm("sxtb16 %0, %1, ror #8" : "=r"(u13) : "r"(u));
    asm("smlad %0, %1, %2, %0" : "+r"(a) : "r"(u02), "r"(xp[k]));
    asm("smlad %0, %1, %2, %0" : "+r"(a) : "r"(u13), "r"(xp[k + 1]));
  }
  return a;
}
#endif

/**
 * int8 x int8 dot product: the ESP32-S3 (esp-nn) or ESP32-P4 (PIE) SIMD
 * kernel when its preconditions hold (16 byte aligned pointers, len a
 * multiple of 16), dotS8Unrolled()
 * otherwise. All give bit-identical results (integer arithmetic).
 */
inline int32_t dotS8(const int8_t* a, const int8_t* b, int len) {
#if defined(TINYSTT_HAS_SIMD_S8)
  if (len >= kSimdAlign && (len % kSimdAlign) == 0 && ((uintptr_t)a % kSimdAlign) == 0 &&
      ((uintptr_t)b % kSimdAlign) == 0) {
#if defined(TINYSTT_P4_PIE)
    return dotS8P4(a, b, len);
#else
    return esp_nn_dot_s8_aligned_esp32s3(a, b, len);
#endif
  }
#endif
#if defined(TINYSTT_HAS_NEON_S8)
  return dotS8Neon(a, b, len);
#elif defined(TINYSTT_HAS_DSP_S8)
  return dotS8Dsp(a, b, len);
#else
  return dotS8Unrolled(a, b, len);
#endif
}

/// true if dotS8() uses a SIMD kernel on this target (esp-nn, PIE or DSP)
inline bool hasSimdS8() {
#if defined(TINYSTT_HAS_SIMD_S8) || defined(TINYSTT_HAS_DSP_S8) || defined(TINYSTT_HAS_NEON_S8)
  return true;
#else
  return false;
#endif
}

/// Name of the int8 dot product kernel used by dotS8()
inline const char* simdS8Name() {
#if defined(TINYSTT_P4_PIE)
  return "ESP32-P4 PIE";
#elif defined(TINYSTT_HAS_SIMD_S8)
  return "ESP32-S3 esp-nn";
#elif defined(TINYSTT_HAS_NEON_S8) && defined(__ARM_FEATURE_DOTPROD)
  return "ARM NEON (SDOT)";
#elif defined(TINYSTT_HAS_NEON_S8)
  return "ARM NEON";
#elif defined(TINYSTT_HAS_DSP_S8)
  return "ARM DSP (SMLAD)";
#else
  return "portable";
#endif
}

}  // namespace tinystt
