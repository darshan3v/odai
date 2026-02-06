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
  DBType m_db_type;
  /// Path to the database file (for SQLite) or connection string (for other backends).
  /// Must be a full file system path for SQLite. Content URIs (e.g., Android content:// URIs) are not supported.
  const char* m_db_path;
};

/// C-style configuration for backend engine (LLM runtime).
/// Used for C API compatibility. Specifies which LLM backend to use.
struct c_BackendEngineConfig
{
  /// Backend engine type to use (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engine_type;
};

/// C-style configuration structure for embedding models.
/// Used for C API compatibility. Contains the path to the embedding model file.
struct c_EmbeddingModelConfig
{
  /// Name of the Embedding model (must be registered).
  c_ModelName m_model_name;
};

inline void free_members(c_EmbeddingModelConfig* config)
{
  if (config == nullptr)
    return;
  if (config->m_model_name)
  {
    free(const_cast<char*>(config->m_model_name));
    config->m_model_name = nullptr;
  }
}

/// C-style configuration structure for language models (LLMs).
/// Used for C API compatibility. Contains the path to the language model file.
struct c_LlmModelConfig
{
  /// Name of the language model (must be registered).
  c_ModelName m_model_name;
};

/// C-style configuration for Fixed Size Chunking Strategy
struct c_FixedSizeChunkingConfig
{
  uint32_t m_chunk_size;
  uint32_t m_chunk_overlap;
};

/// C-style configuration for Chunking Strategy
struct c_ChunkingConfig
{
  ChunkingStrategy m_strategy;
  union
  {
    struct c_FixedSizeChunkingConfig m_fixed_size_config;
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
  struct c_EmbeddingModelConfig m_embedding_model_config;
  struct c_ChunkingConfig m_chunking_config;
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
  free_members(&config->m_embedding_model_config);
  free_members(&config->m_chunking_config);
}

/// C-style configuration structure for Retrieval system.
/// Used for C API compatibility.
struct c_RetrievalConfig
{
  uint32_t m_top_k;
  uint32_t m_fetch_k;
  float m_score_threshold;
  SearchType m_search_type;
  bool m_use_reranker;
  uint32_t m_context_window;
};

/// C-style configuration for RAG Generation (Runtime/Generator use)
struct c_GeneratorRagConfig
{
  struct c_RetrievalConfig m_retrieval_config;
  c_SemanticSpaceName m_semantic_space_name;
  c_ScopeId m_scope_id;
};

/// C-style configuration structure for Sampler (LLM generation parameters).
/// Used for C API compatibility.
struct c_SamplerConfig
{
  uint32_t m_max_tokens;
  float m_top_p;
  uint32_t m_top_k;
};

/// C-style configuration for Generator
struct c_GeneratorConfig
{
  struct c_SamplerConfig m_sampler_config;
  RagMode m_rag_mode;
  // Optional configuration. Null if not used.
  struct c_GeneratorRagConfig* m_rag_config;
};

/// C-style configuration structure for chat sessions.
/// Used for C API compatibility. Defines the behavior and settings for a chat session.
struct c_ChatConfig
{
  /// Whether chat messages should be persisted to the database
  bool m_persistence;
  /// System prompt that defines the assistant's behavior and instructions.
  const char* m_system_prompt;
  /// Configuration for the language model used in this chat session
  struct c_LlmModelConfig m_llm_model_config;
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
  char* m_message_metadata;
  /// Unix timestamp when the message was created
  uint64_t m_created_at;
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

  if (message->m_message_metadata)
  {
    free(message->m_message_metadata);
    message->m_message_metadata = nullptr;
  }
}
