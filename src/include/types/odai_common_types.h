#pragma once

#include <cstdint>

/// Odai Result Types
typedef uint32_t c_OdaiResult;
#define ODAI_SUCCESS (c_OdaiResult)0
#define ODAI_ALREADY_EXISTS (c_OdaiResult)1
#define ODAI_NOT_FOUND (c_OdaiResult)2
#define ODAI_VALIDATION_FAILED (c_OdaiResult)3
#define ODAI_INVALID_ARGUMENT (c_OdaiResult)4
#define ODAI_INTERNAL_ERROR (c_OdaiResult)5
#define ODAI_NOT_INITIALIZED (c_OdaiResult)6

/// Callback function type for streaming response tokens.
/// Called for each chunk generated during streaming response generation.
/// @param token The generated utf-8 resp string to append to the response
/// @param user_data User-provided data pointer passed when calling the streaming function
/// @return true to continue streaming, false to cancel or suspend the streaming output
typedef bool (*OdaiStreamRespCallbackFn)(const char* token, void* user_data);

typedef uint8_t OdaiLogLevel;
#define ODAI_LOG_ERROR (OdaiLogLevel)0
#define ODAI_LOG_WARN (OdaiLogLevel)1
#define ODAI_LOG_INFO (OdaiLogLevel)2
#define ODAI_LOG_DEBUG (OdaiLogLevel)3
#define ODAI_LOG_TRACE (OdaiLogLevel)4

/// Database type identifier for selecting which database backend to use.
typedef uint8_t DBType;
#define SQLITE_DB (DBType)0

/// Backend engine type identifier.
/// Use the constants like LLAMA_BACKEND_ENGINE to specify which backend to use.
typedef uint8_t BackendEngineType;
#define LLAMA_BACKEND_ENGINE (BackendEngineType)0

/// C-style representation of BackendDeviceType
typedef uint8_t c_BackendDeviceType;
#define ODAI_BACKEND_DEVICE_TYPE_CPU ((c_BackendDeviceType)0)
#define ODAI_BACKEND_DEVICE_TYPE_GPU ((c_BackendDeviceType)1)
#define ODAI_BACKEND_DEVICE_TYPE_IGPU ((c_BackendDeviceType)2)
#define ODAI_BACKEND_DEVICE_TYPE_AUTO ((c_BackendDeviceType)3)

/// Strategy for Chunking
typedef uint8_t ChunkingStrategy;
#define FIXED_SIZE_CHUNKING (ChunkingStrategy)0

/// Search Type for Retrieval
typedef uint8_t SearchType;
#define SEARCH_TYPE_VECTOR_ONLY (SearchType)0
#define SEARCH_TYPE_KEYWORD_ONLY (SearchType)1
#define SEARCH_TYPE_HYBRID (SearchType)2

/// RAG Mode
typedef uint8_t RagMode;
#define RAG_MODE_ALWAYS (RagMode)0
#define RAG_MODE_NEVER (RagMode)1
#define RAG_MODE_DYNAMIC (RagMode)2

constexpr uint32_t DEFAULT_CHUNKING_SIZE = 512;
constexpr uint32_t DEFAULT_CHUNKING_OVERLAP = 50;

constexpr uint32_t DEFAULT_MAX_TOKENS = 4096;
constexpr float DEFAULT_TOP_P = 0.95F;
constexpr uint32_t DEFAULT_TOP_K = 40;
constexpr uint32_t DEFAULT_LLM_CONTEXT_WINDOW = 2048;
constexpr uint32_t DEFAULT_EMBEDDING_CONTEXT_WINDOW = 512;

constexpr uint64_t BYTES_PER_KB = 1024ULL;
constexpr uint64_t BYTES_PER_MB = 1024ULL * BYTES_PER_KB;
constexpr uint64_t BYTES_PER_GB = 1024ULL * BYTES_PER_MB;
