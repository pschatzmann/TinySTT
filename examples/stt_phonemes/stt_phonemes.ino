/**
 * Step 1 only: prints the phonemes recognized from an I2S microphone, one
 * line per utterance (e.g. "HH AH L OW W ER L D"). Useful to check the
 * microphone and the acoustic model without the decoder.
 *
 * PhonemeOutput takes the place of the phoneme queue and prints to Serial;
 * PhonemeOutput::Frames prints every frame with its probabilities instead.
 * Only the acoustic model sections of /tinystt.tstt are used.
 */
#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "LittleFS.h"
#include "TinySTT.h"

I2SStream mic;
AudioRealFFT fft;
ModelContainer model;
PhonemeOutput phonemes;                     // prints to Serial
SpeechToPhonemes speechToPhonemes(fft, phonemes);
StreamCopy copier(speechToPhonemes, mic);  // microphone -> step 1

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");
  if (!file || model.begin(file, file.size()) != ModelContainer::OK || !speechToPhonemes.begin(model)) {
    Serial.println("could not load /tinystt.tstt");
    while (true) delay(1000);
  }

  // I2S microphone: 16 kHz, mono, 16 bit (adjust the pins for your board)
  auto cfg = mic.defaultConfig(RX_MODE);
  cfg.sample_rate = 16000;
  cfg.channels = 1;
  cfg.bits_per_sample = 16;
  cfg.pin_bck = 14;
  cfg.pin_ws = 15;
  cfg.pin_data_rx = 32;
  mic.begin(cfg);
}

void loop() { copier.copy(); }
