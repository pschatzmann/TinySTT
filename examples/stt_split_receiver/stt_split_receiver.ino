/**
 * Split setup, device B: Serial1 -> StreamToQueue -> phoneme queue ->
 * PhonemesToText -> Serial. Needs only the lexicon and language model
 * (no audio, no acoustic model). Device A runs stt_split_sender.
 */
#include "AudioTools.h"
#include "LittleFS.h"
#include "TinySTT.h"

PhonemeQueue queue(64);
StreamToQueue receiver;
PhonemesToText phonemesToText(queue);
ModelContainer model;

void setup() {
  Serial.begin(115200);
#if defined(ESP32)
  Serial1.begin(115200, SERIAL_8N1, 18 /*rx*/, 17 /*tx*/);
#else
  Serial1.begin(115200);  // default UART pins of the board
#endif
  AudioToolsLogger.begin(Serial, AudioToolsLogLevel::Warning);

  LittleFS.begin();
  File file = LittleFS.open("/tinystt.tstt", "r");  // lexicon + LM sections
  if (!file || model.begin(file, file.size()) != ModelContainer::OK) {
    Serial.println("could not load /tinystt.tstt");
    while (true) delay(1000);
  }
  phonemesToText.begin(model);
  phonemesToText.setOutput(Serial);  // one line per recognized utterance
  receiver.begin(Serial1, queue);
}

void loop() {
  receiver.copy();        // producer: parses frames from Serial1
  phonemesToText.copy();  // consumer: decodes them
}
