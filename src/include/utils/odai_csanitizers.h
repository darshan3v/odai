#include "types/odai_ctypes.h"

/// Validates that a database configuration is sane and usable.
/// Checks that the database path is not null.
/// @param config The database configuration to validate
/// @return true if the configuration is valid (has a non-null dbPath), false otherwise
inline bool is_sane(const c_DBConfig *config)
{
    return config != nullptr &&
           config->dbPath != nullptr;
}

/// Validates that a backend engine configuration is sane and usable.
/// @param config The backend engine configuration to validate
/// @return true if the configuration is valid, false otherwise
inline bool is_sane(const c_BackendEngineConfig *config)
{
    return config != nullptr;
}

/// Validates that an embedding model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The embedding model configuration to validate
/// @return true if the configuration is valid (has a non-null model path), false otherwise
inline bool is_sane(const c_EmbeddingModelConfig *config)
{
    return config != nullptr &&
           config->modelPath != nullptr;
}

/// Validates that a chunking configuration is sane and usable.
/// @param config The chunking configuration to validate
/// @return true if the configuration is valid
inline bool is_sane(const c_ChunkingConfig *config)
{
    if (config == nullptr)
        return false;

    if (config->strategy != FIXED_SIZE_CHUNKING)
        return false;

    return true;
}

/// Validates that a semantic space configuration is sane.
/// @param config The semantic space configuration to validate
/// @return true of the configuration is valid
inline bool is_sane(const c_SemanticSpaceConfig *config)
{
    return config != nullptr &&
           config->name != nullptr &&
           is_sane(&config->embeddingModelConfig) &&
           is_sane(&config->chunkingConfig);
}

/// Validates that a language model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The language model configuration to validate
/// @return true if the configuration is valid (has a non-null model path), false otherwise
inline bool is_sane(const c_LLMModelConfig *config)
{
    return config != nullptr &&
           config->modelPath != nullptr;
}

/// Validates that a RAG configuration is sane and usable.
/// Checks that both the embedding and LLM model configurations are valid.
/// @param config The RAG configuration to validate
/// @return true if the configuration is valid (has valid embedding and LLM configs), false otherwise
inline bool is_sane(const c_RagConfig *config)
{
    return config != nullptr &&
           is_sane(&config->embeddingModelConfig) &&
           is_sane(&config->llmModelConfig);
}

/// Validates that a chat configuration is sane and usable.
/// Checks that the system prompt is not null, and that the LLM model configuration is valid.
/// @param config The chat configuration to validate
/// @return true if the configuration is valid (has non-null system prompt and valid LLM config), false otherwise
inline bool is_sane(const c_ChatConfig *config)
{
    return config != nullptr &&
           config->system_prompt != nullptr &&
           is_sane(&config->llmModelConfig);
}