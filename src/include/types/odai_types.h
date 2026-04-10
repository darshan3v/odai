#pragma once

#include "odai_ctypes.h"
#include "types/odai_common_types.h"
#include <cstdint>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

// typedefs and enums

/// Strong type for chat session identifiers.
/// Provides type safety for chat session IDs.
typedef std::string ChatId;

/// Strong type for document identifiers.
/// Provides type safety for document IDs.
typedef std::string DocumentId;

/// Strong type for scope identifiers (used for RAG context grouping).
/// Provides type safety for scope IDs.
typedef std::string ScopeId;

/// Strong type for semantic space names.
typedef std::string SemanticSpaceName;

/// Strong type for model names.
typedef std::string ModelName;

enum ModelType : std::uint8_t
{
  EMBEDDING = 0,
  LLM = 1
};

enum UpdateModelFlag : std::uint8_t
{
  STRICT_MATCH = ODAI_UPDATE_STRICT_MATCH,
  ALLOW_MISMATCH = ODAI_UPDATE_ALLOW_MISMATCH
};

enum class OdaiResultEnum : std::uint32_t
{
  ALREADY_EXISTS = ODAI_ALREADY_EXISTS,
  NOT_FOUND = ODAI_NOT_FOUND,
  VALIDATION_FAILED = ODAI_VALIDATION_FAILED,
  INVALID_ARGUMENT = ODAI_INVALID_ARGUMENT,
  INTERNAL_ERROR = ODAI_INTERNAL_ERROR,
  NOT_INITIALIZED = ODAI_NOT_INITIALIZED
};

/// Specifies the desired output format for the decoded audio.
struct OdaiAudioTargetSpec
{
  uint32_t m_sampleRate;
  uint8_t m_channels;
};

/// Holds the raw float32 PCM samples and metadata after decoding.
struct OdaiDecodedAudio
{
  std::vector<float> m_samples;
  uint32_t m_sampleRate{};
  uint8_t m_channels{};
};

/// Specifies the desired output format for the decoded image.
struct OdaiImageTargetSpec
{
  uint32_t m_maxWidth{};  // Maximum allowed width (0 to keep original)
  uint32_t m_maxHeight{}; // Maximum allowed height (0 to keep original)
  uint8_t m_channels{};   // Desired number of channels (e.g., 3 for RGB, 4 for RGBA, 0 to keep original)
};

/// Holds the raw pixel data and metadata after decoding an image.
struct OdaiDecodedImage
{
  std::vector<uint8_t> m_pixels;
  uint32_t m_width{};
  uint32_t m_height{};
  uint8_t m_channels{};
};

struct ModelFiles
{
  ModelType m_modelType;
  BackendEngineType m_engineType{};
  std::unordered_map<std::string, std::string> m_entries;

  // This single line generates both == and !=
  bool operator==(const ModelFiles&) const = default;

  bool is_sane() const { return m_engineType == LLAMA_BACKEND_ENGINE; }
};

/// Internal enum to determine the media class
enum class MediaType : std::uint8_t
{
  TEXT = 0,
  IMAGE = 1,
  AUDIO = 2,
  INVALID = UINT8_MAX
};

/// Represents the type of input item format.
enum class InputItemType : std::uint8_t
{
  FILE_PATH = 0,
  MEMORY_BUFFER = 1,
};

struct InputItem
{
  InputItemType m_type;
  std::vector<uint8_t> m_data;
  std::string m_mimeType;

  /// Identifies the MediaType from a given mime_type string.
  /// @return The appropriate MediaType enum.
  MediaType get_media_type() const
  {
    if (m_mimeType.rfind("text/", 0) == 0)
    {
      return MediaType::TEXT;
    }
    if (m_mimeType.rfind("image/", 0) == 0)
    {
      return MediaType::IMAGE;
    }
    if (m_mimeType.rfind("audio/", 0) == 0)
    {
      return MediaType::AUDIO;
    }

    return MediaType::INVALID;
  }

  bool is_sane() const { return !m_data.empty() && !m_mimeType.empty() && (get_media_type() != MediaType::INVALID); }
};

struct DBConfig
{
  /// Database type to use (SQLITE_DB, POSTGRES_DB, etc.)
  DBType m_dbType{};

  /// Path to the database file (for SQLite) or connection string (for other
  /// backends in future). Must be a full file system path for SQLite. Content URIs (e.g.,
  /// Android content:// URIs) are not supported.
  std::string m_dbPath;

  /// Global absolute path where DB should store media files (e.g. images/audio).
  std::string m_mediaStorePath;

  bool is_sane() const
  {
    if (m_dbPath.empty() || m_mediaStorePath.empty())
    {
      return false;
    }

    if (m_dbType != SQLITE_DB)
    {
      return false;
    }

    return true;
  }
};

enum class BackendDeviceType : std::uint8_t
{
  CPU = ODAI_BACKEND_DEVICE_TYPE_CPU,
  GPU = ODAI_BACKEND_DEVICE_TYPE_GPU,
  IGPU = ODAI_BACKEND_DEVICE_TYPE_IGPU,
  AUTO = ODAI_BACKEND_DEVICE_TYPE_AUTO
};

struct BackendDevice
{
  std::string m_name;
  std::string m_description;
  BackendDeviceType m_type{};
  uint64_t m_totalRam{};
};

/// Summary of a completed streaming generation call.
/// `m_wasCancelled` indicates the user callback stopped the stream early.
struct StreamingStats
{
  int32_t m_generatedTokens{};
  bool m_wasCancelled{};
};

/// Configuration structure for backend engine (LLM runtime).
/// Specifies which LLM backend to use for text generation.
struct BackendEngineConfig
{
  /// Backend engine type (e.g., LLAMA_BACKEND_ENGINE)
  BackendEngineType m_engineType;

  /// Preferred device type (e.g., CPU, GPU, Auto)
  BackendDeviceType m_preferredDeviceType;

  bool is_sane() const { return m_engineType == LLAMA_BACKEND_ENGINE; }
};

/// Configuration structure for embedding models.
/// Contains the path to the embedding model file used for generating vector
/// embeddings.
struct EmbeddingModelConfig
{
  /// Name of the embedding model to use (must be registered via
  /// odai_register_model_files).
  ModelName m_modelName;

  bool is_sane() const { return !m_modelName.empty(); }
};

/// Configuration structure for language models (LLMs).
struct LLMModelConfig
{
  /// Name of the language model to use (must be registered via
  /// odai_register_model_files).
  ModelName m_modelName;

  // This single line generates both == and !=
  bool operator==(const LLMModelConfig&) const = default;

  bool is_sane() const { return !m_modelName.empty(); }
};

/// Configuration for Fixed Size Chunking Strategy
struct FixedSizeChunkingConfig
{
  uint32_t m_chunkSize = DEFAULT_CHUNKING_SIZE;
  uint32_t m_chunkOverlap = DEFAULT_CHUNKING_OVERLAP;

  bool is_sane() const
  {
    if (m_chunkSize == 0)
    {
      return false;
    }
    if (m_chunkOverlap >= m_chunkSize)
    {
      return false;
    }
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
  uint32_t m_dimensions{};

  bool is_sane() const
  {
    if (m_name.empty())
    {
      return false;
    }
    if (!m_embeddingModelConfig.is_sane())
    {
      return false;
    }
    if (!m_chunkingConfig.is_sane())
    {
      return false;
    }
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
    {
      return false;
    }
    if (m_scoreThreshold < 0.0F || m_scoreThreshold > 1.0F)
    {
      return false;
    }

    return true;
  }
};

/// Configuration for RAG Generation (Runtime/Generator use)
/// Uses SemanticSpaceName to reference an existing space.
struct GeneratorRagConfig
{
  RetrievalConfig m_retrievalConfig{};
  SemanticSpaceName m_semanticSpaceName;
  ScopeId m_scopeId;

  bool is_sane() const
  {
    if (!m_retrievalConfig.is_sane())
    {
      return false;
    }
    if (m_semanticSpaceName.empty())
    {
      return false;
    }
    if (m_scopeId.empty())
    {
      return false;
    }
    return true;
  }
};

/// Full Configuration for RAG Generation
/// Includes full SemanticSpaceConfig definition.
struct RagGenerationConfig
{
  RetrievalConfig m_retrievalConfig{};
  SemanticSpaceConfig m_semanticSpaceConfig;

  bool is_sane() const
  {
    if (!m_retrievalConfig.is_sane())
    {
      return false;
    }
    if (!m_semanticSpaceConfig.is_sane())
    {
      return false;
    }
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
    {
      return false;
    }
    if (m_topP < 0.0F || m_topP > 1.0F)
    {
      return false;
    }
    if (m_topK <= 0)
    {
      return false;
    }

    return true;
  }
};

/// Configuration for Generator
struct GeneratorConfig
{
  SamplerConfig m_samplerConfig;
  RagMode m_ragMode{};
  std::optional<GeneratorRagConfig> m_ragConfig;

  bool is_sane() const
  {
    if (!m_samplerConfig.is_sane())
    {
      return false;
    }

    if (m_ragMode == RAG_MODE_NEVER)
    {
      if (m_ragConfig.has_value())
      {
        return false;
      }
    }
    else // ALWAYS or DYNAMIC
    {
      if (!m_ragConfig.has_value())
      {
        return false;
      }
      if (!m_ragConfig->is_sane())
      {
        return false;
      }
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
  bool m_persistence{};
  /// System prompt that defines the assistant's behavior and instructions
  std::string m_systemPrompt;
  /// Configuration for the language model used in this chat session
  LLMModelConfig m_llmModelConfig;

  bool is_sane() const
  {
    if (m_systemPrompt.empty())
    {
      return false;
    }
    if (!m_llmModelConfig.is_sane())
    {
      return false;
    }

    return true;
  }
};

/// Structure representing a chat message.
/// Contains the message role, content, metadata, and creation timestamp.
struct ChatMessage
{
  /// Role of the message sender ('user', 'assistant', or 'system')
  std::string m_role;
  /// The message content items that is the prompt (with / without media items in order)
  std::vector<InputItem> m_contentItems;
  /// JSON object for additional metadata (citations, context, etc.)
  nlohmann::json m_messageMetadata;
  /// Unix timestamp when the message was created
  uint64_t m_createdAt{};

  bool is_sane() const
  {
    if (m_role != "user" && m_role != "assistant" && m_role != "system")
    {
      return false;
    }

    if (m_contentItems.empty())
    {
      return false;
    }

    for (const auto& item : m_contentItems)
    {
      if (!item.is_sane())
      {
        return false;
      }
    }
    return true;
  }
};

struct StreamingBufferContext
{
  std::string m_bufferedResponse;
  OdaiStreamRespCallbackFn m_userCallback{};
  void* m_userData{};
};
