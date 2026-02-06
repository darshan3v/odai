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
typedef const char* c_SemanticSpaceName;

/// Model Name - opaque string type for type safety.
typedef const char* c_ModelName;

/// Model Path - opaque string type for type safety.
typedef const char* c_ModelPath;

/// Model Type - opaque type for model classification
typedef uint32_t c_ModelType;
#define ODAI_MODEL_TYPE_EMBEDDING (c_ModelType)0
#define ODAI_MODEL_TYPE_LLM (c_ModelType)1

struct c_DbConfig
{
  /// Database type to use (SQLITE_DB, etc.)
  DBType m_dbType;
  /// Path to the database file (for SQLite) or connection string (for other backends).
  /// Must be a full file system path for SQLite. Content URIs (e.g., Android content:// URIs) are not supported.
  const char* m_dbPath;
};

/// C-style configuration for backend engine (LLM runtime).
/// Used for C API compatibility. Specifies which LLM backend to use.
struct c_BackendEngineConfig
{
  /// Backend engine type to use (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engineType;
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
    return;
  if (config->m_modelName)
  {
    free(const_cast<char*>(config->m_modelName));
    config->m_modelName = nullptr;
  }
}

/// C-style configuration structure for language models (LLMs).
/// Used for C API compatibility. Contains the path to the language model file.
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
    return;
  if (config->m_name)
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
  /// The message content text (dynamically allocated, caller must free)
  char* m_content;
  /// JSON string containing additional metadata (citations, context, etc.) (dynamically allocated, caller must free)
  char* m_messageMetadata;
  /// Unix timestamp when the message was created
  uint64_t m_createdAt;
};

inline void free_members(c_ChatMessage* message)
{
  if (message == nullptr)
    return;

  if (message->m_content)
  {
    free(message->m_content);
    message->m_content = nullptr;
  }

  if (message->m_messageMetadata)
  {
    free(message->m_messageMetadata);
    message->m_messageMetadata = nullptr;
  }
}
