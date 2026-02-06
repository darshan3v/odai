#include "types/odai_ctypes.h"

inline bool is_sane(c_ModelType type)
{
  return (type == ODAI_MODEL_TYPE_EMBEDDING || type == ODAI_MODEL_TYPE_LLM);
}

/// Validates that a database configuration is sane and usable.
/// Checks that the database path is not null.
/// @param config The database configuration to validate
/// @return true if the configuration is valid (has a non-null dbPath), false
/// otherwise
inline bool is_sane(const c_DbConfig* config)
{
  return config != nullptr && config->m_db_path != nullptr;
}

/// Validates that a backend engine configuration is sane and usable.
/// @param config The backend engine configuration to validate
/// @return true if the configuration is valid, false otherwise
inline bool is_sane(const c_BackendEngineConfig* config)
{
  return config != nullptr;
}

/// Validates that an embedding model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The embedding model configuration to validate
/// @return true if the configuration is valid (has a non-null model path),
/// false otherwise
inline bool is_sane(const c_EmbeddingModelConfig* config)
{
  return config != nullptr && config->m_model_name != nullptr;
}

/// Validates that a language model configuration is sane and usable.
/// Checks that the model path is not null.
/// @param config The language model configuration to validate
/// @return true if the configuration is valid (has a non-null model path),
/// false otherwise
inline bool is_sane(const c_LlmModelConfig* config)
{
  return config != nullptr && config->m_model_name != nullptr;
}

/// Validates that a chunking configuration is sane and usable.
/// @param config The chunking configuration to validate
/// @return true if the configuration is valid
inline bool is_sane(const c_ChunkingConfig* config)
{

  // Here we check strategy is valid because its also structural validation
  // since we are using union if it was some business logic validation then we
  // would have to do it in cpp type is_sane() not here

  if (config == nullptr)
    return false;

  if (config->m_strategy != FIXED_SIZE_CHUNKING)
    return false;

  return true;
}

/// Validates that a semantic space configuration is sane.
/// @param config The semantic space configuration to validate
/// @return true of the configuration is valid
inline bool is_sane(const c_SemanticSpaceConfig* config)
{
  return config != nullptr && config->m_name != nullptr && is_sane(&config->m_embedding_model_config) &&
         is_sane(&config->m_chunking_config);
}

/// Validates that a Retrieval configuration is sane and usable.
/// @param config The Retrieval configuration to validate
/// @return true if the configuration is valid, false otherwise
inline bool is_sane(const c_RetrievalConfig* config)
{
  // Minimal check here, detailed check in C++ type
  return config != nullptr;
}

/// Validates that a Sampler configuration is sane.
/// @param config The Sampler configuration to validate
/// @return true if the configuration is valid
inline bool is_sane(const c_SamplerConfig* config)
{
  return config != nullptr;
}

inline bool is_sane(const struct c_GeneratorRagConfig* config)
{
  if (config == nullptr)
    return false;
  if (!is_sane(&config->m_retrieval_config))
    return false;
  if (config->m_semantic_space_name == nullptr)
    return false;
  if (config->m_scope_id == nullptr)
    return false;
  return true;
}

inline bool is_sane(const struct c_GeneratorConfig* config)
{
  if (config == nullptr)
    return false;
  // Check sampler config sane? Using standard logic inside types if available,
  // otherwise manual check c_SamplerConfig manual check:

  if (config->m_rag_mode == RAG_MODE_NEVER)
  {
    if (config->m_rag_config != nullptr)
      return false;
  }
  else // ALWAYS or DYNAMIC
  {
    if (config->m_rag_config == nullptr)
      return false;
    if (!is_sane(config->m_rag_config))
      return false;
  }

  return true;
}

/// Validates that a chat configuration is sane and usable.
/// Checks that the system prompt is not null, and that the LLM model
/// configuration is valid.
/// @param config The chat configuration to validate
/// @return true if the configuration is valid (has non-null system prompt and
/// valid LLM config), false otherwise
inline bool is_sane(const c_ChatConfig* config)
{
  return config != nullptr && config->m_system_prompt != nullptr && is_sane(&config->m_llm_model_config);
}