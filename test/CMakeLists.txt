# Host-side tests. Normally reached via the top-level CMakeLists.txt:
#   cmake -S . -B build -DTINYSTT_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build
# Every test_*.cpp becomes one executable registered with CTest.
file(GLOB TINYSTT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
foreach(src ${TINYSTT_TEST_SOURCES})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE tinystt_host)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${name} PRIVATE TINYSTT_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME ${name} COMMAND ${name})
endforeach()

# Optional TTS -> STT round trip against the sibling TinyTTS library
# (header-only). Runs only when TINYSTT_MODEL points to a trained container.
option(TINYSTT_ROUNDTRIP "Build the TinyTTS -> TinySTT round trip test" OFF)
set(TINYSTT_TINYTTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../TinyTTS" CACHE PATH "TinyTTS checkout")
if(TINYSTT_ROUNDTRIP AND EXISTS "${TINYSTT_TINYTTS_DIR}/src/TinyTTS.h")
  # TinyTTS expects FS.h when ARDUINO is defined, which the full emulator
  # lacks: use audio-tools' lightweight emulation (as TinyTTS' own CLI does)
  add_executable(test_roundtrip roundtrip_tts.cpp)
  target_link_libraries(test_roundtrip PRIVATE tinystt arduino-audio-tools Threads::Threads ${CMAKE_DL_LIBS} m)
  target_compile_definitions(test_roundtrip PRIVATE IS_MIN_DESKTOP NO_MAIN)
  target_include_directories(test_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TINYSTT_TINYTTS_DIR}/src)
  add_test(NAME test_roundtrip COMMAND test_roundtrip)
endif()
