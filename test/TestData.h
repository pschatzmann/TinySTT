#pragma once
// Helpers to load files from test/data (TINYSTT_TEST_DATA is set by CMake).
#include <fstream>
#include <string>
#include <vector>

inline std::vector<uint8_t> readTestFile(const std::string& name) {
  std::ifstream f(std::string(TINYSTT_TEST_DATA) + "/" + name, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::vector<uint8_t> buf((size_t)f.tellg());
  f.seekg(0);
  f.read((char*)buf.data(), buf.size());
  return buf;
}
