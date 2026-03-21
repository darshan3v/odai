#pragma once

#include "ggml-backend.h"
#include "types/odai_types.h"

/// Convert from GGML backend device type to ODAI BackendDeviceType
BackendDeviceType to_odai_backend_device_type(enum ggml_backend_dev_type type);
