#pragma once

#include "types/odai_common_types.h"
#include <nlohmann/json.hpp>
#include <string>
#include <cstdint>
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
    /// Path to the embedding model file (e.g., .gguf format).
    /// Must be a full file system path. Content URIs (e.g., Android content:// URIs) are not supported.
    string modelPath;

    bool is_sane() const
    {
        if (modelPath.empty())
            return false;

        return true;
    }

};

/// Configuration structure for language models (LLMs).
/// Contains the path to the language model file used for text generation.
struct LLMModelConfig
{
    /// Path to the language model file (e.g., .gguf format).
    /// Must be a full file system path. Content URIs (e.g., Android content:// URIs) are not supported.
    string modelPath;

    bool is_sane() const
    {
        if (modelPath.empty())
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

/// Configuration structure for RAG (Retrieval-Augmented Generation) system.
/// Combines embedding and language model configurations along with RAG-specific settings.
struct RagConfig
{
    /// Configuration for the embedding model used to generate vector embeddings
    EmbeddingModelConfig embeddingModelConfig;
    /// Configuration for the language model used for text generation
    LLMModelConfig llmModelConfig;
    // Additional configuration parameters like strategy etc...
    RagProfile profile;

    bool is_sane() const
    {
        if (!embeddingModelConfig.is_sane() || !llmModelConfig.is_sane())
            return false;
        
        if (profile != RAG_PROFILE_GENERAL && 
            profile != RAG_PROFILE_CODE && 
            profile != RAG_PROFILE_PRECISE && 
            profile != RAG_PROFILE_FAST)
            return false;

        return true;
    }

};

/// Configuration structure for chat sessions.
/// Defines the behavior and settings for a chat session including persistence, RAG usage, and model configuration.
struct ChatConfig
{
    /// Whether chat messages should be persisted to the database
    bool persistence;
    /// Whether to use RAG (Retrieval-Augmented Generation) for this chat session
    bool use_rag;
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

enum ModelType
{
    EMBEDDING,
    LLM
};

struct StreamingBufferContext
{
    string buffered_response;
    odai_stream_resp_callback_fn user_callback;
    void *user_data;
};