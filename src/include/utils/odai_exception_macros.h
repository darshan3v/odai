#pragma once

#include "odai_logger.h"

/// Shared catch tail for exception-to-result translations at boundary functions.
/// Use this only after an explicit `try` block in the caller so logs retain the caller's `__func__` and `__LINE__`.
/// @param fallback_return_expression Expression returned after logging the exception.
#define ODAI_CATCH_RETURN(fallback_return_expression)                                                                  \
  catch (const std::exception& e)                                                                                      \
  {                                                                                                                    \
    ODAI_LOG(ODAI_LOG_ERROR, "{} failed with exception: {}", __func__, e.what());                                      \
    return fallback_return_expression;                                                                                 \
  }                                                                                                                    \
  catch (...)                                                                                                          \
  {                                                                                                                    \
    ODAI_LOG(ODAI_LOG_ERROR, "{} failed with unknown exception", __func__);                                            \
    return fallback_return_expression;                                                                                 \
  }

/// Shared catch tail for exception-to-void translations at boundary functions.
/// Use this only after an explicit `try` block in the caller so logs retain the caller's `__func__` and `__LINE__`.
#define ODAI_CATCH_LOG()                                                                                               \
  catch (const std::exception& e)                                                                                      \
  {                                                                                                                    \
    ODAI_LOG(ODAI_LOG_ERROR, "{} failed with exception: {}", __func__, e.what());                                      \
    return;                                                                                                            \
  }                                                                                                                    \
  catch (...)                                                                                                          \
  {                                                                                                                    \
    ODAI_LOG(ODAI_LOG_ERROR, "{} failed with unknown exception", __func__);                                            \
    return;                                                                                                            \
  }
