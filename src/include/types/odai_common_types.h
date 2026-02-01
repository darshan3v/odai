#pragma once

#include <cstdint>

/// Callback function type for streaming response tokens.
/// Called for each chunk generated during streaming response generation.
/// @param token The generated utf-8 resp string to append to the response
/// @param user_data User-provided data pointer passed when calling the streaming function
/// @return true to continue streaming, false to cancel or suspend the streaming output
typedef bool (*odai_stream_resp_callback_fn)(const char *token, void *user_data);

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

#define DEFAULT_CHUNKING_SIZE 512
#define DEFAULT_CHUNKING_OVERLAP 50

#define DEFAULT_MAX_TOKENS 4096
#define DEFAULT_TOP_P 0.95f
#define DEFAULT_TOP_K 40