/**
 * Speech to text on the Waveshare ESP32-P4-WIFI6: the onboard microphone
 * (ES8311 codec) via audio-tools' AudioBoardStream, recognized utterances
 * printed to Serial.
 *
 * The model is downloaded over Wi-Fi to the TF card with ModelDownloader and
 * audio-tools' URLStream (only once: later starts find it on the card) and
 * loaded from the card into PSRAM. The codec is read as 16 kHz, mono, 16 bit,
 * which TinySTT takes directly. A StreamCopy in loop() runs speech ->
 * phonemes; begin() starts the decoder task on the other core.
 *
 * Needs a FAT32 card in the TF slot; set wifiSSID and wifiPassword.
 * modelURL selects the model: tinystt_large.tstt (9.5 MB, WER 25.7 % at
 * beam 64) or tinystt.tstt (3.1 MB, WER 29.6 % at the default beam 32).
 *
 * Board: "ESP32P4 Dev Module", ChipVariant=postv3 (rev 3.x), PSRAM enabled,
 * Partition Scheme "Huge APP (3MB No OTA/1MB SPIFFS)".
 * Dependencies: arduino-audio-tools, arduino-audio-driver.
 */
#include "AudioTools.h"
#include "AudioTools/AudioLibs/AudioBoardStream.h"
#include "AudioTools/Communication/HTTP/URLStream.h"
#include "AudioTools/FFT/AudioRealFFT.h"
#include "SD_MMC.h"
#include "TinySTT.h"
#include "TinySTT/ModelDownloader.h"

const char* wifiSSID = "ssid";
const char* wifiPassword = "password";
const char* modelURL = "https://github.com/pschatzmann/TinySTT/raw/main/data/tinystt_large.tstt";
const char* modelPath = "/tinystt_large.tstt";
const int beam = 64;  // 64 for the large model, 32 for tinystt.tstt

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

void printProgress(size_t received, size_t total, void*) {
  static size_t next = 0;
  if (received < next) next = 0;
  if (received >= next) {
    Serial.printf("downloaded %u of %u bytes\n", (unsigned)received, (unsigned)total);
    next = received + 1024 * 1024;
  }
}

void setup() {
  Serial.begin(115200);
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  // model: downloaded to the TF card once, then loaded into PSRAM
  if (!SD_MMC.begin("/sdcard", true)) stop("no TF card");  // 1-bit mode, default P4 pins
  if (!ModelDownloader::isValid(SD_MMC, modelPath)) {
    URLStream url(wifiSSID, wifiPassword);  // logs into Wi-Fi
    ModelDownloader downloader(url);
    downloader.setProgressCallback(printProgress);
    uint32_t t0 = millis();
    if (!downloader.download(SD_MMC, modelPath, modelURL)) stop("download failed");
    Serial.printf("download: %u bytes in %u ms\n", (unsigned)downloader.receivedBytes(), (unsigned)(millis() - t0));
  }

  // load the model from the TF card into PSRAM
  File file = SD_MMC.open(modelPath, "r");
  if (!file || model.begin(file, file.size()) != ModelContainer::OK) {
    Serial.printf("could not load %s\n", modelPath);
    stop("model error");
  }
  file.close();

  // decoder config: beam size and word-end beam
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
