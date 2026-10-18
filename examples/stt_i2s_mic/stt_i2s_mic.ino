/**
 * Speech to text from an I2S microphone, printed to Serial.
 *
 * TinySTT is an audio-tools AudioOutput: a StreamCopy in loop() feeds it the
 * microphone data and speech -> phonemes runs there (core 1); begin() starts
 * the decoder task (core 0), connected by a lock-free SPSC queue. Each
 * recognized utterance is printed as one line.
 *
 * The model container (acoustic model + lexicon + LM, see docs/model-data.md)
 * is loaded from LittleFS: upload it as /tinystt.tstt (data/ folder).
 * Board: ESP32-S3 / ESP32-P4 with PSRAM, partition scheme with >= 4 MB FS.
 */
#include "AudioTools.h"
#include "AudioTools/FFT/AudioRealFFT.h"  // any AudioFFTBase works; measured fastest on S3 and P4
#include "LittleFS.h"
#include "TinySTT.h"

I2SStream mic;
AudioRealFFT fft;
ModelContainer model;
TinySTT stt(fft, model, Serial);  // prints one line per utterance
StreamCopy copier(stt, mic);       // microphone -> TinySTT

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  // model: copied into PSRAM
  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");
  if (!file || model.begin(file, file.size()) != ModelContainer::OK) {
    Serial.println("could not load /tinystt.tstt");
    while (true) delay(1000);
  }
  file.close();

  // microphone: 16 kHz, mono, 16 bit (adjust the pins for your board)
  auto cfg = mic.defaultConfig(RX_MODE);
  cfg.sample_rate = 16000;
  cfg.channels = 1;
  cfg.bits_per_sample = 16;
  cfg.i2s_format = I2S_STD_FORMAT;
  cfg.pin_bck = 14;
  cfg.pin_ws = 15;
  cfg.pin_data_rx = 32;
  mic.begin(cfg);

  if (!stt.begin()) {
    Serial.println("TinySTT begin failed");
    while (true) delay(1000);
  }
  Serial.println("listening...");
}

void loop() { copier.copy(); }
