#pragma once

#include "types/odai_common_types.h"
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>

using namespace std;
using namespace nlohmann;

/// Strong type for chat session identifiers.
/// Provides type safety for chat session IDs.
typedef string ChatId;

/// Strong type for document identifiers.
/// Provides type safety for document IDs.
typedef string DocumentId;

/// Strong type for scope identifiers (used for RAG context grouping).
/// Provides type safety for scope IDs.
typedef string ScopeId;

/// Strong type for semantic space names.
typedef string SemanticSpaceName;

/// Strong type for model names.
typedef string ModelName;

/// Strong type for model paths.
typedef string ModelPath;

enum ModelType
{
  EMBEDDING,
  LLM
};

struct DBConfig
{
  /// Database type to use (SQLITE_DB, POSTGRES_DB, etc.)
  DBType m_db_type;
  /// Path to the database file (for SQLite) or connection string (for other
  /// backends). Must be a full file system path for SQLite. Content URIs (e.g.,
  /// Android content:// URIs) are not supported.
  string m_db_path;

  bool is_sane() const
  {
    if (m_db_path.empty())
      return false;

    if (m_db_type != SQLITE_DB)
      return false;

    return true;
  }
};

/// Configuration structure for backend engine (LLM runtime).
/// Specifies which LLM backend to use for text generation.
struct BackendEngineConfig
{
  /// Backend engine type (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engine_type;

  bool is_sane() const
  {
    if (m_engine_type != LLAMA_BACKEND_ENGINE)
      return false;

    return true;
  }
};

/// Configuration structure for embedding models.
/// Contains the path to the embedding model file used for generating vector
/// embeddings.
struct EmbeddingModelConfig
{
  /// Name of the embedding model to use (must be registered via
  /// odai_register_model).
  ModelName m_model_name;

  bool is_sane() const
  {
    if (m_model_name.empty())
      return false;

    return true;
  }
};

/// Configuration structure for language models (LLMs).
/// Contains the path to the language model file used for text generation.
struct LLMModelConfig
{
  /// Name of the language model to use (must be registered via
  /// odai_register_model).
  ModelName m_model_name;

  bool is_sane() const
  {
    if (m_model_name.empty())
      return false;

    return true;
  }
};

/// Configuration for Fixed Size Chunking Strategy
struct FixedSizeChunkingConfig
{
  uint32_t m_chunk_size = DEFAULT_CHUNKING_SIZE;
  uint32_t m_chunk_overlap = DEFAULT_CHUNKING_OVERLAP;

  bool is_sane() const
  {
    if (m_chunk_size == 0)
      return false;
    if (m_chunk_overlap >= m_chunk_size)
      return false;
    return true;
  }
};

/// Configuration for Chunking Strategy
/// Contains the strategy type and union of specific configuration parameters.
struct ChunkingConfig
{
  std::variant<FixedSizeChunkingConfig> m_config;

  bool is_sane() const
  {
    if (std::holds_alternative<FixedSizeChunkingConfig>(m_config))
    {
      const auto& conf = std::get<FixedSizeChunkingConfig>(m_config);
      return conf.is_sane();
    }

    return false;
  }

  ChunkingConfig() { m_config = FixedSizeChunkingConfig(); }
};

/// Configuration structure for Semantic Space.
/// Defines the Embedding Model, Chunking Strategy, and Embedding Dimensions.
struct SemanticSpaceConfig
{
  SemanticSpaceName m_name;
  EmbeddingModelConfig m_embedding_model_config;
  ChunkingConfig m_chunking_config;
  uint32_t m_dimensions;

  bool is_sane() const
  {
    if (m_name.empty())
      return false;
    if (!m_embedding_model_config.is_sane())
      return false;
    if (!m_chunking_config.is_sane())
      return false;
    // dimensions == 0 means auto-infer from model

    return true;
  }
};

/// Configuration structure for Retrieval (RAG) system.
/// Defines the search strategy and parameters for retrieving context.
struct RetrievalConfig
{
  /// How many maximum final chunks to give the LLM?
  uint32_t m_top_k;
  /// How many maximum candidates to fetch initially (before reranking)?
  uint32_t m_fetch_k;
  /// Minimum similarity score (0.0 to 1.0). Discard irrelevant noise.
  float m_score_threshold;
  /// Search Strategy type (VECTOR_ONLY, KEYWORD_ONLY, or HYBRID).
  SearchType m_search_type;
  /// Should we run a cross-encoder? (Expensive but accurate)
  bool m_use_reranker;
  /// If Chunk 5 is a hit, do we also grab Chunk 4 and 6?
  uint32_t m_context_window;

  bool is_sane() const
  {
    if (m_top_k == 0)
      return false;
    if (m_score_threshold < 0.0f || m_score_threshold > 1.0f)
      return false;

    return true;
  }
};

/// Configuration for RAG Generation (Runtime/Generator use)
/// Uses SemanticSpaceName to reference an existing space.
struct GeneratorRagConfig
{
  RetrievalConfig m_retrieval_config;
  SemanticSpaceName m_semantic_space_name;
  ScopeId m_scope_id;

  bool is_sane() const
  {
    if (!m_retrieval_config.is_sane())
      return false;
    if (m_semantic_space_name.empty())
      return false;
    if (m_scope_id.empty())
      return false;
    return true;
  }
};

/// Full Configuration for RAG Generation
/// Includes full SemanticSpaceConfig definition.
struct RagGenerationConfig
{
  RetrievalConfig m_retrieval_config;
  SemanticSpaceConfig m_semantic_space_config;

  bool is_sane() const
  {
    if (!m_retrieval_config.is_sane())
      return false;
    if (!m_semantic_space_config.is_sane())
      return false;
    return true;
  }
};

/// Configuration structure for Sampler (LLM generation parameters).
/// Defines token limits, and sampling strategies.
struct SamplerConfig
{
  uint32_t m_max_tokens = DEFAULT_MAX_TOKENS;
  float m_top_p = DEFAULT_TOP_P;
  uint32_t m_top_k = DEFAULT_TOP_K;

  bool is_sane() const
  {
    if (m_max_tokens == 0)
      return false;
    if (m_top_p < 0.0f || m_top_p > 1.0f)
      return false;
    if (m_top_k <= 0)
      return false;

    return true;
  }
};

/// Configuration for Generator
struct GeneratorConfig
{
  SamplerConfig m_sampler_config;
  RagMode m_rag_mode;
  std::optional<GeneratorRagConfig> m_rag_config;

  bool is_sane() const
  {
    if (!m_sampler_config.is_sane())
      return false;

    if (m_rag_mode == RAG_MODE_NEVER)
    {
      if (m_rag_config.has_value())
        return false;
    }
    else // ALWAYS or DYNAMIC
    {
      if (!m_rag_config.has_value())
        return false;
      if (!m_rag_config->is_sane())
        return false;
    }

    return true;
  }
};

/// Configuration structure for chat sessions.

/// Defines the behavior and settings for a chat session including persistence,
/// RAG usage, and model configuration.
struct ChatConfig
{
  /// Whether chat messages should be persisted to the database
  bool m_persistence;
  /// System prompt that defines the assistant's behavior and instructions
  string m_system_prompt;
  /// Configuration for the language model used in this chat session
  LLMModelConfig m_llm_model_config;

  bool is_sane() const
  {
    if (m_system_prompt.empty())
      return false;
    if (!m_llm_model_config.is_sane())
      return false;

    return true;
  }
};

/// Structure representing a chat message.
/// Contains the message role, content, metadata, and creation timestamp.
struct ChatMessage
{
  /// Role of the message sender ('user', 'assistant', or 'system')
  string m_role;
  /// The message content text
  string m_content;
  /// JSON object for additional metadata (citations, context, etc.)
  json m_message_metadata;
  /// Unix timestamp when the message was created
  uint64_t m_created_at;

  bool is_sane() const
  {
    if (m_role != "user" && m_role != "assistant" && m_role != "system")
      return false;
    if (m_content.empty())
      return false;

    return true;
  }
};

struct StreamingBufferContext
{
  string m_buffered_response;
  OdaiStreamRespCallbackFn m_user_callback;
  void* m_user_data;
};