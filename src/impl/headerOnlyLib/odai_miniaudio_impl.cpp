#define MINIAUDIO_IMPLEMENTATION
// Keep all miniaudio-generated symbols local to this translation unit.
#define MA_API static
// Work around the unprefixed global lock symbol emitted by miniaudio.
#define ma_atomic_global_lock odai_ma_atomic_global_lock
#include "miniaudio.h"
#undef ma_atomic_global_lock
#undef MA_API
