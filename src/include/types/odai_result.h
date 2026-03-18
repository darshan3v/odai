#pragma once

#include "types/odai_types.h"
#include <tl/expected.hpp>

/// C++ Result type that explicitly defines whether an operation succeeded along with its return value,
/// or failed with a specific OdaiResultEnum error code.
template <class T>
using OdaiResult = tl::expected<T, OdaiResultEnum>;
