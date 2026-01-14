#pragma once

#ifdef BUILDING_ODAI_SHARED
#define ODAI_API __attribute__((visibility("default")))
#else
#define ODAI_API
#endif
