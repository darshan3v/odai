#pragma once

#include "odai_common_types.h"
#include <stdlib.h>

/// Chat session identifier - opaque string type for type safety.
typedef char* c_ChatId;

/// Document identifier - opaque string type for type safety.
typedef char* c_DocumentId;

/// Scope identifier - opaque string type for type safety (used for RAG context grouping).
typedef char* c_ScopeId;

/// Semantic Space Name - opaque string type for type safety.c
typedef char* c_SemanticSpaceName;

/// Model Name - opaque string type for type safety.
typedef char* c_ModelName;

/// Model Type - opaque type for model classification
typedef uint32_t c_ModelType;
#define ODAI_MODEL_TYPE_EMBEDDING (c_ModelType)0
#define ODAI_MODEL_TYPE_LLM (c_ModelType)1

/// Flags for updating model registration details
typedef uint32_t c_UpdateModelFlag;
#define ODAI_UPDATE_STRICT_MATCH ((c_UpdateModelFlag)0)
#define ODAI_UPDATE_ALLOW_MISMATCH ((c_UpdateModelFlag)1)

/// Key-Value entry for model registration files
struct c_ModelFileEntry
{
  const char* m_key;
  const char* m_value;
};

/// Generic struct to hold model registration files
struct c_ModelFiles
{
  c_ModelType m_modelType;
  BackendEngineType m_engineType;
  struct c_ModelFileEntry* m_entries;
  size_t m_entriesCount;
};

inline void free_members(c_ModelFiles* model_file_details)
{
  if (model_file_details == nullptr)
  {
    return;
  }
  if (model_file_details->m_entries != nullptr)
  {
    for (size_t i = 0; i < model_file_details->m_entriesCount; ++i)
    {
      if (model_file_details->m_entries[i].m_key != nullptr)
      {
        free(const_cast<char*>(model_file_details->m_entries[i].m_key));
      }
      if (model_file_details->m_entries[i].m_value != nullptr)
      {
        free(const_cast<char*>(model_file_details->m_entries[i].m_value));
      }
    }
    free(model_file_details->m_entries);
    model_file_details->m_entries = nullptr;
  }
}

/// Input Item Type for Multimodal Support
typedef uint8_t c_InputItemType;
#define ODAI_INPUT_ITEM_TYPE_FILE_PATH ((c_InputItemType)0)
#define ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER ((c_InputItemType)1)
#define ODAI_INPUT_ITEM_TYPE_INVALID ((c_InputItemType)UINT8_MAX)

/// C-style structure representing a single input item for generation or chat.
struct c_InputItem
{
  c_InputItemType m_type;
  /// Pointer to the data buffer. For text, it's a character string. For binary, it's bytes.
  void* m_data;
  /// Size of the data in bytes.
  size_t m_dataSize;
  /// Optional MIME type (e.g., "image/jpeg"). Dynamically allocated if present, caller frees.
  char* m_mimeType;
};

inline void free_members(c_InputItem* item)
{
  if (item == nullptr)
  {
    return;
  }

  if (item->m_data != nullptr)
  {
    free(item->m_data);
    item->m_data = nullptr;
  }

  if (item->m_mimeType != nullptr)
  {
    free(item->m_mimeType);
    item->m_mimeType = nullptr;
  }
}

struct c_DbConfig
{
  /// Database type to use (SQLITE_DB, etc.)
  DBType m_dbType;
  /// Path to the database file (for SQLite) or connection string (for other backends).
  /// Must be a full file system path for SQLite. Content URIs (e.g., Android content:// URIs) are not supported.
  const char* m_dbPath;
  /// Global absolute path where DB should store media files (e.g. images/audio)
  const char* m_mediaStorePath;
};

/// C-style configuration for backend engine (LLM runtime).
/// Used for C API compatibility. Specifies which LLM backend to use.
struct c_BackendEngineConfig
{
  /// Backend engine type to use (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engineType;

  /// Preferred device type (e.g., CPU, GPU, Auto)
  c_BackendDeviceType m_preferredDeviceType;
};

/// C-style configuration structure for embedding models.
/// Used for C API compatibility. Contains the path to the embedding model file.
struct c_EmbeddingModelConfig
{
  /// Name of the Embedding model (must be registered).
  c_ModelName m_modelName;
};

inline void free_members(c_EmbeddingModelConfig* config)
{
  if (config == nullptr)
  {
    return;
  }
  if (config->m_modelName != nullptr)
  {
    free(const_cast<char*>(config->m_modelName));
    config->m_modelName = nullptr;
  }
}

/// C-style configuration structure for language models (LLMs).
/// Used for C API compatibility.
struct c_LlmModelConfig
{
  /// Name of the language model (must be registered).
  c_ModelName m_modelName;
};

/// C-style configuration for Fixed Size Chunking Strategy
struct c_FixedSizeChunkingConfig
{
  uint32_t m_chunkSize;
  uint32_t m_chunkOverlap;
};

/// C-style configuration for Chunking Strategy
struct c_ChunkingConfig
{
  ChunkingStrategy m_strategy;
  union
  {
    struct c_FixedSizeChunkingConfig m_fixedSizeConfig;
  } m_config;
};

inline void free_members(c_ChunkingConfig* config)
{
  // No dynamic memory currently
}

/// C-style configuration structure for Semantic Space.
struct c_SemanticSpaceConfig
{
  c_SemanticSpaceName m_name;
  struct c_EmbeddingModelConfig m_embeddingModelConfig;
  struct c_ChunkingConfig m_chunkingConfig;
  uint32_t m_dimensions;
};

inline void free_members(c_SemanticSpaceConfig* config)
{
  if (config == nullptr)
  {
    return;
  }
  if (config->m_name != nullptr)
  {
    free(const_cast<char*>(config->m_name));
    config->m_name = nullptr;
  }
  free_members(&config->m_embeddingModelConfig);
  free_members(&config->m_chunkingConfig);
}

/// C-style configuration structure for Retrieval system.
/// Used for C API compatibility.
struct c_RetrievalConfig
{
  uint32_t m_topK;
  uint32_t m_fetchK;
  float m_scoreThreshold;
  SearchType m_searchType;
  bool m_useReranker;
  uint32_t m_contextWindow;
};

/// C-style configuration for RAG Generation (Runtime/Generator use)
struct c_GeneratorRagConfig
{
  struct c_RetrievalConfig m_retrievalConfig;
  c_SemanticSpaceName m_semanticSpaceName;
  c_ScopeId m_scopeId;
};

/// C-style configuration structure for Sampler (LLM generation parameters).
/// Used for C API compatibility.
struct c_SamplerConfig
{
  uint32_t m_maxTokens;
  float m_topP;
  uint32_t m_topK;
};

/// C-style configuration for Generator
struct c_GeneratorConfig
{
  struct c_SamplerConfig m_samplerConfig;
  RagMode m_ragMode;
  // Optional configuration. Null if not used.
  struct c_GeneratorRagConfig* m_ragConfig;
};

/// C-style configuration structure for chat sessions.
/// Used for C API compatibility. Defines the behavior and settings for a chat session.
struct c_ChatConfig
{
  /// Whether chat messages should be persisted to the database
  bool m_persistence;
  /// System prompt that defines the assistant's behavior and instructions.
  const char* m_systemPrompt;
  /// Configuration for the language model used in this chat session
  struct c_LlmModelConfig m_llmModelConfig;
};

/// C-style structure for chat messages.
/// Used for C API compatibility. Contains message role, content, metadata, and timestamp.
struct c_ChatMessage
{
  /// Role of the message sender ('user', 'assistant', or 'system')
  char m_role[32];
  /// Array of input items making up the message content (dynamically allocated, caller must free)
  struct c_InputItem* m_contentItems;
  /// Number of items in the array
  size_t m_contentItemsCount;
  /// JSON string containing additional metadata (citations, context, etc.) (dynamically allocated, caller must free)
  char* m_messageMetadata;
  /// Unix timestamp when the message was created
  uint64_t m_createdAt;
};

inline void free_members(c_ChatMessage* message)
{
  if (message == nullptr)
  {
    return;
  }

  if (message->m_contentItems != nullptr)
  {
    for (size_t i = 0; i < message->m_contentItemsCount; ++i)
    {
      free_members(&message->m_contentItems[i]);
    }
    free(message->m_contentItems);
    message->m_contentItems = nullptr;
  }

  if (message->m_messageMetadata != nullptr)
  {
    free(message->m_messageMetadata);
    message->m_messageMetadata = nullptr;
  }
}
