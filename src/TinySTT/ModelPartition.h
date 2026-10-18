#pragma once
#if defined(ESP32)
#include "esp_partition.h"
#include "TinySTT/ModelContainer.h"

namespace tinystt {

/**
 * @brief ESP32: memory-maps a model container stored in a flash data
 * partition, so the weights, lexicon and LM are used in place (zero copy,
 * read through the flash cache) instead of being loaded into RAM.
 *
 * partitions.csv:   tinystt, data, 0x40, , 8M
 * flash:            esptool write-flash <partition offset> tinystt.tstt
 */
class ModelPartition {
 public:
  ~ModelPartition() { end(); }

  /// Maps the container of partition `label` into `mc`. The CRC check reads
  /// the whole container once (slow for large files); it is off by default.
  bool begin(ModelContainer& mc, const char* label = "tinystt", bool verifyCrc = false) {
    end();
    const esp_partition_t* part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (part == nullptr) {
      LOGE("TinySTT: partition '%s' not found", label);
      return false;
    }
    uint8_t header[ModelContainer::kHeaderSize];
    if (esp_partition_read(part, 0, header, sizeof(header)) != ESP_OK) return false;
    uint32_t total = ModelContainer::rd32(header + 8);
    if (ModelContainer::rd32(header) != ModelContainer::kMagic || total > part->size) {
      LOGE("TinySTT: no model container in partition '%s'", label);
      return false;
    }
    const void* ptr = nullptr;
    if (esp_partition_mmap(part, 0, total, ESP_PARTITION_MMAP_DATA, &ptr, &handle) != ESP_OK) {
      LOGE("TinySTT: mmap of %u bytes failed", (unsigned)total);
      return false;
    }
    mapped = true;
    return mc.begin((const uint8_t*)ptr, total, verifyCrc) == ModelContainer::OK;
  }

  void end() {
    if (mapped) esp_partition_munmap(handle);
    mapped = false;
  }

 protected:
  esp_partition_mmap_handle_t handle = 0;
  bool mapped = false;
};

}  // namespace tinystt
#endif
