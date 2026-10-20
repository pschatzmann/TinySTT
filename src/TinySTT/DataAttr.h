#pragma once

/**
 * @brief Storage attribute for the generated model-data arrays in
 * TinySTT/data/ (see research/export_header.py): PROGMEM on Arduino, so
 * cores with a separate constant-data area put them there, and empty
 * elsewhere (on ESP32 and RP2040 PROGMEM is itself empty: const data stays
 * in flash). Define TINYSTT_PROGMEM before including a data header to
 * override it.
 */
#ifndef TINYSTT_PROGMEM
#ifdef ARDUINO
#include <Arduino.h>
#endif
#ifdef PROGMEM
#define TINYSTT_PROGMEM PROGMEM
#else
#define TINYSTT_PROGMEM
#endif
#endif
