/**
 * Speech to text on the Waveshare ESP32-P4-WIFI6: the onboard microphone
 * (ES8311 codec) via audio-tools' AudioBoardStream, recognized utterances
 * printed to Serial.
 *
 * The codec is read as 16 kHz, mono, 16 bit, which TinySTT takes directly.
 * The model container is compiled into the program (PROGMEM
 * array, 3.1 MB) and used in place from flash: no file system and no RAM
 * copy. A StreamCopy in loop() runs speech -> phonemes; begin() starts the
 * decoder task on the other core.
 *
 * Board: "ESP32P4 Dev Module", ChipVariant=postv3 (rev 3.x), PSRAM enabled,
 * Flash Size 32MB, Partition Scheme "32M Flash (13MB APP/6.75MB SPIFFS)".
 * Dependencies: arduino-audio-tools, arduino-audio-driver.
 */
#include "AudioTools.h"
#include "AudioTools/AudioLibs/AudioBoardStream.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "TinySTT.h"
#include "TinySTT/data/tinystt_model_data.h"  // the shipped model as PROGMEM array

AudioInfo micInfo(16000, 1, 16);  // the format TinySTT expects
AudioBoardStream board(WaveshareESP32P4WIFI6);
AudioRealFFT fft;
ModelContainer model;
TinySTT stt(fft, model, Serial);  // prints one line per utterance
StreamCopy copier(stt, board);     // microphone -> TinySTT

void stop(const char* msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  // model in place from flash
  if (model.begin(tinystt_model_data, tinystt_model_data_len) != ModelContainer::OK) stop("invalid model data");

  // onboard microphone
  auto cfg = board.defaultConfig(RX_MODE);
  cfg.copyFrom(micInfo);
  if (!board.begin(cfg)) stop("could not start the codec");
  board.setInputVolume(0.8);  // microphone gain, 0..1

  if (!stt.begin()) stop("TinySTT begin failed");
  Serial.println("listening...");
}

void loop() { copier.copy(); }
