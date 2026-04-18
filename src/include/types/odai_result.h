#pragma once

#include "types/odai_types.h"
#include <tl/expected.hpp>

/// C++ Result type that explicitly defines whether an operation succeeded along with its return value,
/// or failed with a specific OdaiResultEnum error code.
template <class T>
using OdaiResult = tl::expected<T, OdaiResultEnum>;

/// Converts a C++ OdaiResultEnum code to the C ABI c_OdaiResult code.
/// This is a value-preserving cast because both enums share the same stable numeric constants.
constexpr c_OdaiResult to_c_result(OdaiResultEnum result)
{
  return static_cast<c_OdaiResult>(result);
}

/// Creates a standard INTERNAL_ERROR unexpected result for catch-all failure paths.
inline tl::unexpected<OdaiResultEnum> unexpected_internal_error()
{
  return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
}

/// Creates a standard NOT_INITIALIZED unexpected result for lifecycle guard paths.
inline tl::unexpected<OdaiResultEnum> unexpected_not_initialized()
{
  return tl::unexpected(OdaiResultEnum::NOT_INITIALIZED);
}
