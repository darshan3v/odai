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
  DBType m_dbType;
  /// Path to the database file (for SQLite) or connection string (for other
  /// backends). Must be a full file system path for SQLite. Content URIs (e.g.,
  /// Android content:// URIs) are not supported.
  string m_dbPath;

  bool is_sane() const
  {
    if (m_dbPath.empty())
      return false;

    if (m_dbType != SQLITE_DB)
      return false;

    return true;
  }
};

/// Configuration structure for backend engine (LLM runtime).
/// Specifies which LLM backend to use for text generation.
struct BackendEngineConfig
{
  /// Backend engine type (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engineType;

  bool is_sane() const
  {
    if (m_engineType != LLAMA_BACKEND_ENGINE)
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
  ModelName m_modelName;

  bool is_sane() const
  {
    if (m_modelName.empty())
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
  ModelName m_modelName;

  bool is_sane() const
  {
    if (m_modelName.empty())
      return false;

    return true;
  }
};

/// Configuration for Fixed Size Chunking Strategy
struct FixedSizeChunkingConfig
{
  uint32_t m_chunkSize = DEFAULT_CHUNKING_SIZE;
  uint32_t m_chunkOverlap = DEFAULT_CHUNKING_OVERLAP;

  bool is_sane() const
  {
    if (m_chunkSize == 0)
      return false;
    if (m_chunkOverlap >= m_chunkSize)
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
  EmbeddingModelConfig m_embeddingModelConfig;
  ChunkingConfig m_chunkingConfig;
  uint32_t m_dimensions;

  bool is_sane() const
  {
    if (m_name.empty())
      return false;
    if (!m_embeddingModelConfig.is_sane())
      return false;
    if (!m_chunkingConfig.is_sane())
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
  uint32_t m_topK;
  /// How many maximum candidates to fetch initially (before reranking)?
  uint32_t m_fetchK;
  /// Minimum similarity score (0.0 to 1.0). Discard irrelevant noise.
  float m_scoreThreshold;
  /// Search Strategy type (VECTOR_ONLY, KEYWORD_ONLY, or HYBRID).
  SearchType m_searchType;
  /// Should we run a cross-encoder? (Expensive but accurate)
  bool m_useReranker;
  /// If Chunk 5 is a hit, do we also grab Chunk 4 and 6?
  uint32_t m_contextWindow;

  bool is_sane() const
  {
    if (m_topK == 0)
      return false;
    if (m_scoreThreshold < 0.0f || m_scoreThreshold > 1.0f)
      return false;

    return true;
  }
};

/// Configuration for RAG Generation (Runtime/Generator use)
/// Uses SemanticSpaceName to reference an existing space.
struct GeneratorRagConfig
{
  RetrievalConfig m_retrievalConfig;
  SemanticSpaceName m_semanticSpaceName;
  ScopeId m_scopeId;

  bool is_sane() const
  {
    if (!m_retrievalConfig.is_sane())
      return false;
    if (m_semanticSpaceName.empty())
      return false;
    if (m_scopeId.empty())
      return false;
    return true;
  }
};

/// Full Configuration for RAG Generation
/// Includes full SemanticSpaceConfig definition.
struct RagGenerationConfig
{
  RetrievalConfig m_retrievalConfig;
  SemanticSpaceConfig m_semanticSpaceConfig;

  bool is_sane() const
  {
    if (!m_retrievalConfig.is_sane())
      return false;
    if (!m_semanticSpaceConfig.is_sane())
      return false;
    return true;
  }
};

/// Configuration structure for Sampler (LLM generation parameters).
/// Defines token limits, and sampling strategies.
struct SamplerConfig
{
  uint32_t m_maxTokens = DEFAULT_MAX_TOKENS;
  float m_topP = DEFAULT_TOP_P;
  uint32_t m_topK = DEFAULT_TOP_K;

  bool is_sane() const
  {
    if (m_maxTokens == 0)
      return false;
    if (m_topP < 0.0f || m_topP > 1.0f)
      return false;
    if (m_topK <= 0)
      return false;

    return true;
  }
};

/// Configuration for Generator
struct GeneratorConfig
{
  SamplerConfig m_samplerConfig;
  RagMode m_ragMode;
  std::optional<GeneratorRagConfig> m_ragConfig;

  bool is_sane() const
  {
    if (!m_samplerConfig.is_sane())
      return false;

    if (m_ragMode == RAG_MODE_NEVER)
    {
      if (m_ragConfig.has_value())
        return false;
    }
    else // ALWAYS or DYNAMIC
    {
      if (!m_ragConfig.has_value())
        return false;
      if (!m_ragConfig->is_sane())
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
  string m_systemPrompt;
  /// Configuration for the language model used in this chat session
  LLMModelConfig m_llmModelConfig;

  bool is_sane() const
  {
    if (m_systemPrompt.empty())
      return false;
    if (!m_llmModelConfig.is_sane())
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
  json m_messageMetadata;
  /// Unix timestamp when the message was created
  uint64_t m_createdAt;

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
  string m_bufferedResponse;
  OdaiStreamRespCallbackFn m_userCallback;
  void* m_userData;
};