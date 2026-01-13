#include "types/odai_ctypes.h"

/// Validates that a database configuration is sane and usable.
/// Checks that the database path is not null.
/// @param config The database configuration to validate
/// @return true if the configuration is valid (has a non-empty dbPath), false otherwise
bool is_sane(const c_DBConfig *config);

/// Validates that a backend engine configuration is sane and usable.
/// Currently checks that a backend engine type is specified.
/// @param config The backend engine configuration to validate
/// @return true if the configuration is valid, false otherwise
bool is_sane(const c_BackendEngineConfig *config);

/// Validates that an embedding model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The embedding model configuration to validate
/// @return true if the configuration is valid (has a non-empty model path), false otherwise
bool is_sane(const c_EmbeddingModelConfig *config);

/// Validates that a language model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The language model configuration to validate
/// @return true if the configuration is valid (has a non-empty model path), false otherwise
bool is_sane(const c_LLMModelConfig *config);

/// Validates that a RAG configuration is sane and usable.
/// Checks that both the embedding and LLM model configurations are valid.
/// @param config The RAG configuration to validate
/// @return true if the configuration is valid (has valid embedding and LLM configs), false otherwise
bool is_sane(const c_RagConfig *config);

/// Validates that a chat configuration is sane and usable.
/// Checks that the system prompt is not null, and that the LLM model configuration is valid.
/// @param config The chat configuration to validate
/// @return true if the configuration is valid (has non-empty system prompt and valid LLM config), false otherwise
bool is_sane(const c_ChatConfig *config);