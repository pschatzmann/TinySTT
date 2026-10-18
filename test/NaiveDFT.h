#pragma once
// Test-only FFT implementation: a naive O(N^2) DFT behind the audio-tools
// FFTDriver interface. Used to show that FeatureExtractor works with any
// injected AudioFFTBase, and as an independent reference for AudioRealFFT.
#include <cmath>
#include <vector>

#include "AudioTools.h"
#include "AudioTools/FFT/AudioFFT.h"

class FFTDriverNaiveDFT : public audio_tools::FFTDriver {
 public:
  bool begin(int len) override {
    n = len;
    x.assign(len, 0.0f);
    re.assign(len / 2 + 1, 0.0);
    im.assign(len / 2 + 1, 0.0);
    return true;
  }
  void end() override {}
  void setValue(int pos, float value) override { x[pos] = value; }
  void fft() override {
    for (int k = 0; k <= n / 2; k++) {
      double r = 0, i = 0;
      for (int t = 0; t < n; t++) {
        double a = -2.0 * M_PI * k * t / n;
        r += x[t] * std::cos(a);
        i += x[t] * std::sin(a);
      }
      re[k] = r;
      im[k] = i;
    }
  }
  float magnitude(int idx) override { return std::sqrt(magnitudeFast(idx)); }
  float magnitudeFast(int idx) override { return (float)(re[idx] * re[idx] + im[idx] * im[idx]); }
  bool isValid() override { return n > 0; }
  float getValue(int pos) override { return x[pos]; }

 protected:
  int n = 0;
  std::vector<float> x;
  std::vector<double> re, im;
};

class AudioNaiveDFT : public audio_tools::AudioFFTBase {
 public:
  AudioNaiveDFT() : AudioFFTBase(new FFTDriverNaiveDFT()) {}
};
