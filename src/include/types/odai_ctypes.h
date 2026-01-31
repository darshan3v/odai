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

struct c_DBConfig
{
    /// Database type to use (SQLITE_DB, etc.)
    DBType dbType;
    /// Path to the database file (for SQLite) or connection string (for other backends).
    /// Must be a full file system path for SQLite. Content URIs (e.g., Android content:// URIs) are not supported.
    const char *dbPath;
};

/// C-style configuration for backend engine (LLM runtime).
/// Used for C API compatibility. Specifies which LLM backend to use.
struct c_BackendEngineConfig
{
    /// Backend engine type to use (e.g., LLAMA_BACKEND_ENGINE)
    BackendEngineType engineType;
};

/// C-style configuration structure for embedding models.
/// Used for C API compatibility. Contains the path to the embedding model file.
struct c_EmbeddingModelConfig
{
    /// Path to the embedding model file (e.g., .gguf format).
    /// Must be a full file system path. Content URIs (e.g., Android content:// URIs) are not supported.
    const char *modelPath;

};


inline void free_members(c_EmbeddingModelConfig* config)
{
    if (config == nullptr) return;
    if (config->modelPath)
    {
        free(const_cast<char *>(config->modelPath));
        config->modelPath = nullptr;
    }
}

/// C-style configuration structure for language models (LLMs).
/// Used for C API compatibility. Contains the path to the language model file.
struct c_LLMModelConfig
{
    /// Path to the language model file (e.g., .gguf format).
    /// Must be a full file system path. Content URIs (e.g., Android content:// URIs) are not supported.
    const char *modelPath;
};

/// C-style configuration structure for RAG (Retrieval-Augmented Generation) system.
/// Used for C API compatibility. Combines embedding and language model configurations.
struct c_RagConfig
{
    /// Configuration for the embedding model used to generate vector embeddings
    struct c_EmbeddingModelConfig embeddingModelConfig;
    /// Configuration for the language model used for text generation
    struct c_LLMModelConfig llmModelConfig;
    /// RAG Profile
    RagProfile profile;
};

/// C-style configuration for Fixed Size Chunking Strategy
struct c_FixedSizeChunkingConfig
{
    uint32_t chunkSize;
    uint32_t chunkOverlap;
};

/// C-style configuration for Chunking Strategy
struct c_ChunkingConfig
{
    ChunkingStrategy strategy;
    union
    {
        struct c_FixedSizeChunkingConfig fixedSizeConfig;
    } config;

};


inline void free_members(c_ChunkingConfig* config)
{
    // No dynamic memory currently
}

/// C-style configuration structure for Semantic Space.
struct c_SemanticSpaceConfig
{
    c_SemanticSpaceName name;
    struct c_EmbeddingModelConfig embeddingModelConfig;
    struct c_ChunkingConfig chunkingConfig;
    uint32_t dimensions;

};


inline void free_members(c_SemanticSpaceConfig* config)
{
    if (config == nullptr) return;
    if (config->name)
    {
        free(const_cast<char *>(config->name));
        config->name = nullptr;
    }
    free_members(&config->embeddingModelConfig);
    free_members(&config->chunkingConfig);
}

/// C-style configuration structure for chat sessions.
/// Used for C API compatibility. Defines the behavior and settings for a chat session.
struct c_ChatConfig
{
    /// Whether chat messages should be persisted to the database
    bool persistence;
    /// Whether to use RAG (Retrieval-Augmented Generation) for this chat session
    bool use_rag;
    /// System prompt that defines the assistant's behavior and instructions.
    const char* system_prompt;
    /// Configuration for the language model used in this chat session
    struct c_LLMModelConfig llmModelConfig;
    // sampler params and scopeid can probably be here but also be overidable during chat 
};

/// C-style structure for chat messages.
/// Used for C API compatibility. Contains message role, content, metadata, and timestamp.
struct c_ChatMessage
{
    /// Role of the message sender ('user', 'assistant', or 'system')
    char role[32];
    /// The message content text (dynamically allocated, caller must free)
    char* content;
    /// JSON string containing additional metadata (citations, context, etc.) (dynamically allocated, caller must free)
    char* message_metadata;
    /// Unix timestamp when the message was created
    uint64_t created_at;
};

inline void free_members(c_ChatMessage* message)
{
    if (message == nullptr) return;
    
    if (message->content)
    {
        free(message->content);
        message->content = nullptr;
    }
    
    if (message->message_metadata)
    {
        free(message->message_metadata);
        message->message_metadata = nullptr;
    }
}
