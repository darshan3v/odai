#pragma once

#include "types/odai_common_types.h"
#include <nlohmann/json.hpp>
#include <string>
#include <cstdint>
#include <variant>
#include <optional>

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
    DBType dbType;
    /// Path to the database file (for SQLite) or connection string (for other backends).
    /// Must be a full file system path for SQLite. Content URIs (e.g., Android content:// URIs) are not supported.
    string dbPath;

    bool is_sane() const
    {
        if (dbPath.empty())
            return false;

        if (dbType != SQLITE_DB)
            return false;

        return true;
    }
};

/// Configuration structure for backend engine (LLM runtime).
/// Specifies which LLM backend to use for text generation.
struct BackendEngineConfig
{
    /// Backend engine type (e.g., LLAMA_BACKEND_ENGINE)
    BackendEngineType engineType;

    bool is_sane() const
    {
        if (engineType != LLAMA_BACKEND_ENGINE)
            return false;

        return true;
    }
};

/// Configuration structure for embedding models.
/// Contains the path to the embedding model file used for generating vector embeddings.
struct EmbeddingModelConfig
{
    /// Name of the embedding model to use (must be registered via odai_register_model).
    ModelName modelName;

    bool is_sane() const
    {
        if (modelName.empty())
            return false;

        return true;
    }

};

/// Configuration structure for language models (LLMs).
/// Contains the path to the language model file used for text generation.
struct LLMModelConfig
{
    /// Name of the language model to use (must be registered via odai_register_model).
    ModelName modelName;

    bool is_sane() const
    {
        if (modelName.empty())
            return false;

        return true;
    }

};

/// Configuration for Fixed Size Chunking Strategy
struct FixedSizeChunkingConfig
{
    uint32_t chunkSize = DEFAULT_CHUNKING_SIZE;
    uint32_t chunkOverlap = DEFAULT_CHUNKING_OVERLAP;

    bool is_sane() const
    {
        if (chunkSize == 0)
            return false;
        if (chunkOverlap >= chunkSize)
            return false;
        return true;
    }
};

/// Configuration for Chunking Strategy
/// Contains the strategy type and union of specific configuration parameters.
struct ChunkingConfig
{
    std::variant<FixedSizeChunkingConfig> config;

    bool is_sane() const
    {
        if (std::holds_alternative<FixedSizeChunkingConfig>(config))
        {
            const auto& conf = std::get<FixedSizeChunkingConfig>(config);
            return conf.is_sane();
        }

        return false;
    }

    ChunkingConfig()
    {
        config = FixedSizeChunkingConfig();
    }
    
};

/// Configuration structure for Semantic Space.
/// Defines the Embedding Model, Chunking Strategy, and Embedding Dimensions.
struct SemanticSpaceConfig
{
    SemanticSpaceName name;
    EmbeddingModelConfig embeddingModelConfig;
    ChunkingConfig chunkingConfig;
    uint32_t dimensions;

    bool is_sane() const
    {
        if (name.empty())
            return false;
        if (!embeddingModelConfig.is_sane())
            return false;
        if (!chunkingConfig.is_sane())
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
    uint32_t top_k;
    /// How many maximum candidates to fetch initially (before reranking)?
    uint32_t fetch_k;
    /// Minimum similarity score (0.0 to 1.0). Discard irrelevant noise.
    float score_threshold;
    /// Search Strategy type (VECTOR_ONLY, KEYWORD_ONLY, or HYBRID).
    SearchType search_type;
    /// Should we run a cross-encoder? (Expensive but accurate)
    bool use_reranker;
    /// If Chunk 5 is a hit, do we also grab Chunk 4 and 6?
    uint32_t context_window;

    bool is_sane() const
    {
        if (top_k == 0)
            return false;
        if (score_threshold < 0.0f || score_threshold > 1.0f)
            return false;
            
        return true;
    }

};

/// Configuration for RAG Generation (Runtime/Generator use)
/// Uses SemanticSpaceName to reference an existing space.
struct GeneratorRagConfig
{
    RetrievalConfig retrievalConfig;
    SemanticSpaceName semanticSpaceName;

    bool is_sane() const
    {
        if (!retrievalConfig.is_sane())
            return false;
        if (semanticSpaceName.empty())
            return false;
        return true;
    }
};

/// Full Configuration for RAG Generation
/// Includes full SemanticSpaceConfig definition.
struct RagGenerationConfig
{
    RetrievalConfig retrievalConfig;
    SemanticSpaceConfig semanticSpaceConfig;

    bool is_sane() const
    {
        if (!retrievalConfig.is_sane())
            return false;
        if (!semanticSpaceConfig.is_sane())
            return false;
        return true;
    }
};

/// Configuration structure for Sampler (LLM generation parameters).
/// Defines token limits, and sampling strategies.
struct SamplerConfig
{
    uint32_t max_tokens = DEFAULT_MAX_TOKENS;
    float top_p = DEFAULT_TOP_P;
    uint32_t top_k = DEFAULT_TOP_K;

    bool is_sane() const
    {
        if (max_tokens == 0)
            return false;
        if (top_p < 0.0f || top_p > 1.0f)
            return false;
        if (top_k <= 0)
            return false;
        
        return true;
    }
};

/// Configuration for Generator
struct GeneratorConfig
{
    SamplerConfig samplerConfig;
    RagMode ragMode;
    std::optional<GeneratorRagConfig> ragConfig;

    bool is_sane() const
    {
        if (!samplerConfig.is_sane())
            return false;

        if (ragMode == RAG_MODE_NEVER)
        {
            if (ragConfig.has_value())
                return false;
        }
        else // ALWAYS or DYNAMIC
        {
            if (!ragConfig.has_value())
                return false;
            if (!ragConfig->is_sane())
                return false;
        }

        return true;
    }
};

/// Configuration structure for chat sessions.

/// Defines the behavior and settings for a chat session including persistence, RAG usage, and model configuration.
struct ChatConfig
{
    /// Whether chat messages should be persisted to the database
    bool persistence;
    /// System prompt that defines the assistant's behavior and instructions
    string system_prompt;
    /// Configuration for the language model used in this chat session
    LLMModelConfig llmModelConfig;

    bool is_sane() const
    {
        if (system_prompt.empty())
            return false;
        if (!llmModelConfig.is_sane())
            return false;

        return true;
    }

};

/// Structure representing a chat message.
/// Contains the message role, content, metadata, and creation timestamp.
struct ChatMessage
{
    /// Role of the message sender ('user', 'assistant', or 'system')
    string role;
    /// The message content text
    string content;
    /// JSON object for additional metadata (citations, context, etc.)
    json message_metadata;
    /// Unix timestamp when the message was created
    uint64_t created_at;

    bool is_sane() const
    {
        if ((role != "user") || (role != "assistant") || (role != "system"))
            return false;
        if (content.empty())
            return false;

        return true;
    }

};

struct StreamingBufferContext
{
    string buffered_response;
    odai_stream_resp_callback_fn user_callback;
    void *user_data;
};