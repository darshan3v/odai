#pragma once

#include <cstdint>

#define ODAI_LOG_ERROR (OdaiLogLevel)0
#define ODAI_LOG_WARN (OdaiLogLevel)1
#define ODAI_LOG_INFO (OdaiLogLevel)2
#define ODAI_LOG_DEBUG (OdaiLogLevel)3
#define ODAI_LOG_TRACE (OdaiLogLevel)4

#define LLAMA_BACKEND_ENGINE (BackendEngineType)0

#define SQLITE_DB (DBType)0

/// Callback function type for streaming response tokens.
/// Called for each chunk generated during streaming response generation.
/// @param token The generated utf-8 resp string to append to the response
/// @param user_data User-provided data pointer passed when calling the streaming function
/// @return true to continue streaming, false to cancel or suspend the streaming output
typedef bool (*odai_stream_resp_callback_fn)(const char *token, void *user_data);

/// Database type identifier for selecting which database backend to use.
typedef uint8_t DBType;

/// Backend engine type identifier.
/// Use the constants like LLAMA_BACKEND_ENGINE to specify which backend to use.
typedef uint8_t BackendEngineType;