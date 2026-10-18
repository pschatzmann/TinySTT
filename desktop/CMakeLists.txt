# Desktop CLI: ./build/desktop/tinystt --model model.tstt speech.wav
#   cmake -S . -B build -DTINYSTT_BUILD_DESKTOP=ON && cmake --build build
# Optional live microphone input via audio-tools' MiniAudioStream:
#   -DTINYSTT_DESKTOP_MIC=ON (fetches the single-header mackron/miniaudio)
add_executable(tinystt_cli main.cpp)
set_target_properties(tinystt_cli PROPERTIES OUTPUT_NAME tinystt)
# A plain desktop program with argc/argv: built against audio-tools'
# lightweight emulation (IS_MIN_DESKTOP) like TinyTTS' CLI -- the
# Arduino-Emulator's API declares `int main()` itself.
target_link_libraries(tinystt_cli PRIVATE tinystt arduino-audio-tools Threads::Threads ${CMAKE_DL_LIBS} m)
target_compile_definitions(tinystt_cli PRIVATE IS_MIN_DESKTOP NO_MAIN)

option(TINYSTT_DESKTOP_MIC "Microphone input for the desktop CLI (miniaudio)" OFF)
if(TINYSTT_DESKTOP_MIC)
  include(FetchContent)
  FetchContent_Declare(miniaudio GIT_REPOSITORY "https://github.com/mackron/miniaudio.git" GIT_TAG master)
  FetchContent_GetProperties(miniaudio)
  if(NOT miniaudio_POPULATED)
    FetchContent_Populate(miniaudio)
  endif()
  target_include_directories(tinystt_cli PRIVATE ${miniaudio_SOURCE_DIR})
  target_compile_definitions(tinystt_cli PRIVATE TINYSTT_HAVE_MIC)
  target_link_libraries(tinystt_cli PRIVATE atomic)
endif()

# Desktop counterpart of examples/timing_benchmark (docs/performance.md)
add_executable(tinystt_benchmark benchmark.cpp)
target_link_libraries(tinystt_benchmark PRIVATE tinystt arduino-audio-tools Threads::Threads ${CMAKE_DL_LIBS} m)
target_compile_definitions(tinystt_benchmark PRIVATE IS_MIN_DESKTOP NO_MAIN)

install(TARGETS tinystt_cli RUNTIME DESTINATION bin)
