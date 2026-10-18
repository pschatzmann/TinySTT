# TinySTT is header-only. This file serves three consumers:
#   - ESP-IDF: detected via ESP_PLATFORM, registers as an IDF component.
#   - Plain CMake (host tests, desktop CLI): INTERFACE library TinySTT::TinySTT.
#   - Arduino: does not use this file -- see library.properties.
if(ESP_PLATFORM)
  idf_component_register(INCLUDE_DIRS "src" REQUIRES arduino-audio-tools)
  return()
endif()

cmake_minimum_required(VERSION 3.16)
project(TinySTT CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# On ARM Linux (Raspberry Pi) the distribution compiler may target the
# lowest common CPU (32 bit Raspberry Pi OS: ARMv6, no NEON); compile the
# host targets for the build machine's CPU instead (NEON int8 kernel).
option(TINYSTT_NATIVE "Host builds: optimize for the build machine's ARM CPU (-mcpu=native)" ON)
if(TINYSTT_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" TINYSTT_HAS_MCPU_NATIVE)
  if(TINYSTT_HAS_MCPU_NATIVE)
    add_compile_options(-mcpu=native)
  endif()
endif()

option(TINYSTT_BUILD_TESTS "Build the host-side tests (test/)" OFF)
option(TINYSTT_BUILD_DESKTOP "Build the desktop CLI (desktop/)" OFF)
# ON: link the full pschatzmann/Arduino-Emulator (IS_DESKTOP).
# OFF: audio-tools' lightweight built-in emulation only (IS_MIN_DESKTOP).
option(TINYSTT_USE_ARDUINO_EMULATOR "Build host targets against the Arduino-Emulator" ON)
set(TINYSTT_SANITIZE "none" CACHE STRING "Sanitizer for host tests: none, address or thread")

add_library(tinystt INTERFACE)
add_library(TinySTT::TinySTT ALIAS tinystt)
target_include_directories(tinystt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(tinystt INTERFACE cxx_std_17)

if(TINYSTT_BUILD_TESTS OR TINYSTT_BUILD_DESKTOP)
  # arduino-audio-tools: prefer a sibling checkout (Arduino libraries folder),
  # otherwise fetch it. The emulator itself is fetched by audio-tools' own
  # CMakeLists.txt (ADD_ARDUINO_EMULATOR); point
  # FETCHCONTENT_SOURCE_DIR_ARDUINO_EMULATOR at a local checkout to avoid
  # the download.
  set(ADD_ARDUINO_EMULATOR ${TINYSTT_USE_ARDUINO_EMULATOR} CACHE BOOL "" FORCE)
  set(ADD_PORTAUDIO OFF CACHE BOOL "" FORCE)
  set(TINYSTT_AUDIO_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../arduino-audio-tools"
      CACHE PATH "Local arduino-audio-tools checkout")
  if(EXISTS "${TINYSTT_AUDIO_TOOLS_DIR}/CMakeLists.txt")
    add_subdirectory(${TINYSTT_AUDIO_TOOLS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/arduino-audio-tools)
  else()
    include(FetchContent)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
    FetchContent_Declare(arduino-audio-tools
      GIT_REPOSITORY "https://github.com/pschatzmann/arduino-audio-tools.git"
      GIT_TAG main)
    FetchContent_MakeAvailable(arduino-audio-tools)
  endif()

  # Host flavour of the library, shared by test/ and desktop/.
  add_library(tinystt_host INTERFACE)
  target_link_libraries(tinystt_host INTERFACE tinystt arduino-audio-tools)
  find_package(Threads REQUIRED)
  target_link_libraries(tinystt_host INTERFACE Threads::Threads ${CMAKE_DL_LIBS} m)
  if(TINYSTT_USE_ARDUINO_EMULATOR)
    target_compile_definitions(tinystt_host INTERFACE IS_DESKTOP)
    target_link_libraries(tinystt_host INTERFACE arduino_emulator)
  else()
    target_compile_definitions(tinystt_host INTERFACE IS_MIN_DESKTOP NO_MAIN)
  endif()
  if(TINYSTT_SANITIZE STREQUAL "address" OR TINYSTT_SANITIZE STREQUAL "thread")
    target_compile_options(tinystt_host INTERFACE -fsanitize=${TINYSTT_SANITIZE} -g -O1)
    target_link_options(tinystt_host INTERFACE -fsanitize=${TINYSTT_SANITIZE})
  endif()
endif()

if(TINYSTT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(TINYSTT_BUILD_DESKTOP)
  add_subdirectory(desktop)
endif()
