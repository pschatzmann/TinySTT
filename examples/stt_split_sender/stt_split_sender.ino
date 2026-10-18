/**
 * Split setup, device A: microphone -> SpeechToPhonemes -> phoneme queue ->
 * QueueToStream -> Serial1. Only the acoustic model is needed on this device.
 * Device B runs stt_split_receiver (PhonemesToText).
 *
 * The phoneme stream is small (~16 bytes per 30 ms frame, much less with
 * blank skipping), so a plain UART at 115200 baud is plenty.
 */
#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "LittleFS.h"
#include "TinySTT.h"

I2SStream mic;
AudioRealFFT fft;
PhonemeQueue queue(64);
SpeechToPhonemes speechToPhonemes(fft, queue);
QueueToStream sender;
ModelContainer model;
StreamCopy copier(speechToPhonemes, mic);  // SpeechToPhonemes is an AudioOutput

void setup() {
  Serial.begin(115200);
#if defined(ESP32)
  Serial1.begin(115200, SERIAL_8N1, 18 /*rx*/, 17 /*tx*/);
#else
  Serial1.begin(115200);  // default UART pins of the board
#endif
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");  // the acoustic model is enough
  speechToPhonemes.setBatchFrames(4);  // less memory traffic, +90 ms latency
  if (!file || model.begin(file, file.size()) != ModelContainer::OK || !speechToPhonemes.begin(model)) {
    Serial.println("could not load /tinystt.tstt");
    while (true) delay(1000);
  }

  auto cfg = mic.defaultConfig(RX_MODE);
  cfg.sample_rate = 16000;
  cfg.channels = 1;
  cfg.bits_per_sample = 16;
  cfg.pin_bck = 14;
  cfg.pin_ws = 15;
  cfg.pin_data_rx = 32;
  mic.begin(cfg);

  sender.begin(queue, Serial1);
}

void loop() {
  copier.copy();   // producer of the queue (speech -> phonemes)
  sender.copy();   // consumer of the queue (frames -> Serial1)
}
