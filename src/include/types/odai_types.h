#pragma once

#include "types/odai_common_types.h"
#include <nlohmann/json.hpp>
#include <string>
#include <cstdint>

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

typedef struct
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
} DBConfig;

/// Configuration structure for backend engine (LLM runtime).
/// Specifies which LLM backend to use for text generation.
typedef struct
{
    /// Backend engine type (e.g., LLAMA_BACKEND_ENGINE)
    BackendEngineType engineType;

    bool is_sane() const
    {
        if (engineType != LLAMA_BACKEND_ENGINE)
            return false;

        return true;
    }
} BackendEngineConfig;

/// Configuration structure for embedding models.
/// Contains the path to the embedding model file used for generating vector embeddings.
typedef struct
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

} EmbeddingModelConfig;

/// Configuration structure for language models (LLMs).
/// Contains the path to the language model file used for text generation.
typedef struct
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

} LLMModelConfig;

/// Configuration structure for RAG (Retrieval-Augmented Generation) system.
/// Combines embedding and language model configurations along with RAG-specific settings.
typedef struct
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

} RagConfig;

/// Configuration structure for chat sessions.
/// Defines the behavior and settings for a chat session including persistence, RAG usage, and model configuration.
typedef struct
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

} ChatConfig;

/// Structure representing a chat message.
/// Contains the message role, content, metadata, and creation timestamp.
typedef struct
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

} ChatMessage;

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