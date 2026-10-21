#pragma once
#include <FS.h>

#include "AudioTools/Communication/HTTP/AbstractURLStream.h"
#include "TinySTT/ModelContainer.h"

namespace tinystt {

/**
 * @brief Downloads a model container with an audio-tools URL stream (e.g.
 * URLStream, which also logs into Wi-Fi) into a file on any Arduino file
 * system (SD, SD_MMC, LittleFS).
 *
 * A valid file that is already there is kept, so the download happens only
 * once. The data goes to `<path>.part` first; the file is renamed to `path`
 * only when its size matches the HTTP content length and the container
 * header, so an interrupted download never leaves a broken model behind.
 * ModelContainer::begin(File&, size) then checks the CRC when it loads it.
 *
 * @code
 * URLStream url("ssid", "password");
 * ModelDownloader downloader(url);
 * downloader.download(SD_MMC, "/tinystt.tstt", "https://.../tinystt.tstt");
 * @endcode
 */
class ModelDownloader {
 public:
  /// called during the download: bytes received and the total (0: unknown)
  using ProgressCallback = void (*)(size_t received, size_t total, void* ref);

  explicit ModelDownloader(audio_tools::AbstractURLStream& url) : p_url(&url) {}

  void setProgressCallback(ProgressCallback cb, void* ref = nullptr) {
    progress = cb;
    progress_ref = ref;
  }
  /// gives up when no data arrives for this long (default 15 s)
  void setTimeout(uint32_t ms) { timeout_ms = ms; }

  /// true if `path` holds a complete container (header magic and size)
  static bool isValid(fs::FS& fs, const char* path) {
    File f = fs.open(path, "r");
    if (!f) return false;
    uint8_t header[ModelContainer::kHeaderSize];
    bool ok = f.read(header, sizeof(header)) == sizeof(header) &&
              ModelContainer::rd32(header) == ModelContainer::kMagic &&
              ModelContainer::rd32(header + 8) == f.size();
    f.close();
    return ok;
  }

  /// Downloads `url` to `path` unless a valid container is already there (or
  /// `force`); returns true if `path` holds a complete container afterwards
  bool download(fs::FS& fs, const char* path, const char* url, bool force = false) {
    if (!force && isValid(fs, path)) {
      LOGI("TinySTT: %s is already there", path);
      return true;
    }
    String part = String(path) + ".part";
    bool ok = fetch(fs, part.c_str(), url) && isValid(fs, part.c_str());
    if (!ok) {
      LOGE("TinySTT: download of %s failed", url);
      fs.remove(part.c_str());
      return false;
    }
    fs.remove(path);
    if (!fs.rename(part.c_str(), path)) {
      LOGE("TinySTT: could not rename %s", part.c_str());
      return false;
    }
    LOGI("TinySTT: downloaded %s (%u bytes)", path, (unsigned)received);
    return true;
  }

  /// bytes received by the last download
  size_t receivedBytes() const { return received; }

 protected:
  audio_tools::AbstractURLStream* p_url;
  ProgressCallback progress = nullptr;
  void* progress_ref = nullptr;
  uint32_t timeout_ms = 15000;
  size_t received = 0;

  bool fetch(fs::FS& fs, const char* path, const char* url) {
    received = 0;
    if (!p_url->begin(url, "application/octet-stream")) {
      LOGE("TinySTT: could not open %s", url);
      return false;
    }
    int len = p_url->contentLength();
    size_t total = len > 0 ? (size_t)len : 0;
    File out = fs.open(path, "w");
    if (!out) {
      LOGE("TinySTT: could not create %s", path);
      p_url->end();
      return false;
    }
    uint8_t buffer[4096];
    uint32_t last_data = millis();
    while (total == 0 || received < total) {
      size_t n = p_url->readBytes(buffer, sizeof(buffer));
      if (n == 0) {
        // without a content length the end is reached when no data follows
        if (millis() - last_data > (total == 0 ? 2000 : timeout_ms)) break;
        delay(1);
        continue;
      }
      if (out.write(buffer, n) != n) {
        LOGE("TinySTT: write to %s failed (file system full?)", path);
        break;
      }
      received += n;
      last_data = millis();
      if (progress) progress(received, total, progress_ref);
    }
    out.close();
    p_url->end();
    return received > 0 && (total == 0 || received == total);
  }
};

}  // namespace tinystt
