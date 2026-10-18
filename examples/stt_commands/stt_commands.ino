/**
 * Voice commands with a grammar: only phrases of the grammar are recognized,
 * which is much more robust than free dictation. Results below 60%
 * confidence are not printed. Settings are made before begin().
 */
#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "LittleFS.h"
#include "TinySTT.h"

I2SStream mic;
AudioRealFFT fft;
ModelContainer model;
TinySTT stt(fft, model, Serial);  // prints one line per utterance
StreamCopy copier(stt, mic);       // microphone -> TinySTT

const char* grammar =
    "(please)? turn (on | off) the (kitchen | bedroom | living room)? (light | lights | fan);"
    "set a timer for (one | two | five | ten) minutes;"
    "(stop | cancel)";

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);
  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");
  if (!file || model.begin(file, file.size()) != ModelContainer::OK) {
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

  if (!stt.phonemesToText().setGrammar(grammar)) {
    Serial.printf("grammar error: %s\n", stt.phonemesToText().grammar().error().c_str());
  }
  stt.phonemesToText().decoderConfig().min_confidence = 0.6f;
  stt.begin();
}

void loop() { copier.copy(); }
