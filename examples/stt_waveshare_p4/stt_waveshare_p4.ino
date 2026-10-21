/**
 * Speech to text on the Waveshare ESP32-P4-WIFI6: the onboard microphone
 * (ES8311 codec) via audio-tools' AudioBoardStream, recognized utterances
 * printed to Serial.
 *
 * The large model (data/tinystt_large.tstt, 9.5 MB, 50k words, WER 25.7 % at
 * beam 64) is compiled into the program and used in place from flash: no
 * file system, no load step, no RAM copy (see docs/model-embedding.md).
 *
 * For the default model (3.1 MB, beam 32) include
 * "TinySTT/data/tinystt_model.h" and use tinystt_model / tinystt_model_size().
 *
 * The codec is read as 16 kHz, mono, 16 bit, which TinySTT takes directly.
 * A StreamCopy in loop() runs speech -> phonemes; begin() starts the decoder
 * task on the other core.
 *
 * Board: "ESP32P4 Dev Module", ChipVariant=postv3 (rev 3.x), PSRAM enabled,
 * Flash Size 32MB, Partition Scheme "32M Flash (13MB APP/6.75MB SPIFFS)".
 * Dependencies: arduino-audio-tools, arduino-audio-driver.
 */

#include "TinySTT.h"
#include "TinySTT/data/tinystt_large_model.h"

#include "AudioTools.h"
#include "AudioTools/AudioLibs/AudioBoardStream.h"
#include "AudioTools/FFT/AudioRealFFT.h"

const int beam = 64;  // 64 for the large model, 32 for tinystt_model

AudioInfo micInfo(16000, 1, 16);  // the format TinySTT expects
AudioBoardStream board(WaveshareESP32P4WIFI6);
AudioRealFFT fft;
ModelContainer model;
TinySTT stt(fft, model, Serial0);  // prints one line per utterance
StreamCopy copier(stt, board);     // microphone -> TinySTT

void stop(const char* msg) {
  Serial.println(msg);
  while (true) delay(1000);
}

void setup() {
  Serial0.begin(115200);
  AudioToolsLogger.begin(Serial0, AudioToolsLogLevel::Warning);

  // model: used in place from flash
  if (model.begin(tinystt_large_model, tinystt_large_model_size()) != ModelContainer::OK) stop("invalid model data");
  stt.phonemesToText().decoderConfig().beam_size = beam;
  stt.phonemesToText().decoderConfig().word_end_beam = beam / 4;

  // onboard microphone
  auto cfg = board.defaultConfig(RX_MODE);
  cfg.copyFrom(micInfo);
  if (!board.begin(cfg)) stop("could not start the codec");
  board.setInputVolume(0.8);  // microphone gain, 0..1

  // start the decoder task on the other core
  if (!stt.begin()) stop("TinySTT begin failed");
  Serial.println("listening...");
}

void loop() { copier.copy(); }
