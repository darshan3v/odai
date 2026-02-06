#include "types/odai_type_conversions.h"

#include <cstring>

ModelType to_cpp(c_ModelType c)
{
  if (c == ODAI_MODEL_TYPE_LLM)
    return ModelType::LLM;
  else if (c == ODAI_MODEL_TYPE_EMBEDDING)
    return ModelType::EMBEDDING;

  // Default to LLM if unknown, or handle error appropriately.
  // Since we can't easily return error here, we assume valid input or handle at caller.
  return ModelType::LLM;
}

DBConfig to_cpp(const c_DbConfig& c)
{
  return {c.m_db_type, string(c.m_db_path)};
}

BackendEngineConfig to_cpp(const c_BackendEngineConfig& c)
{
  return {c.m_engine_type};
}

EmbeddingModelConfig to_cpp(const c_EmbeddingModelConfig& c)
{
  return {string(c.m_model_name)};
}

LLMModelConfig to_cpp(const c_LlmModelConfig& c)
{
  return {string(c.m_model_name)};
}

ChunkingConfig to_cpp(const c_ChunkingConfig& c)
{
  ChunkingConfig config;

  if (c.m_strategy == FIXED_SIZE_CHUNKING)
  {
    FixedSizeChunkingConfig fcc;
    fcc.m_chunk_size = c.m_config.m_fixed_size_config.m_chunk_size;
    fcc.m_chunk_overlap = c.m_config.m_fixed_size_config.m_chunk_overlap;
    config.m_config = fcc;
  }
  return config;
}

SemanticSpaceConfig to_cpp(const c_SemanticSpaceConfig& c)
{
  SemanticSpaceConfig config;
  config.m_name = string(c.m_name);
  config.m_embedding_model_config = to_cpp(c.m_embedding_model_config);
  config.m_chunking_config = to_cpp(c.m_chunking_config);
  config.m_dimensions = c.m_dimensions;
  return config;
}

RetrievalConfig to_cpp(const c_RetrievalConfig& c)
{
  RetrievalConfig config;
  config.m_top_k = c.m_top_k;
  config.m_fetch_k = c.m_fetch_k;
  config.m_score_threshold = c.m_score_threshold;
  config.m_search_type = c.m_search_type;
  config.m_use_reranker = c.m_use_reranker;
  config.m_context_window = c.m_context_window;
  return config;
}

SamplerConfig to_cpp(const c_SamplerConfig& c)
{
  return {c.m_max_tokens, c.m_top_p, c.m_top_k};
}

GeneratorRagConfig to_cpp(const c_GeneratorRagConfig& source)
{
  GeneratorRagConfig config;
  config.m_retrieval_config = to_cpp(source.m_retrieval_config);
  if (source.m_semantic_space_name)
  {
    config.m_semantic_space_name = string(source.m_semantic_space_name);
  }
  if (source.m_scope_id)
  {
    config.m_scope_id = string(source.m_scope_id);
  }
  return config;
}

GeneratorConfig to_cpp(const c_GeneratorConfig& source)
{
  GeneratorConfig config;
  config.m_sampler_config = to_cpp(source.m_sampler_config);
  config.m_rag_mode = source.m_rag_mode;

  if (source.m_rag_config != nullptr)
  {
    config.m_rag_config = to_cpp(*source.m_rag_config);
  }
  else
  {
    config.m_rag_config = std::nullopt;
  }

  return config;
}

ChatConfig to_cpp(const c_ChatConfig& c)
{
  return {c.m_persistence, string(c.m_system_prompt), to_cpp(c.m_llm_model_config)};
}

c_EmbeddingModelConfig to_c(const EmbeddingModelConfig& cpp)
{
  c_EmbeddingModelConfig c;
  c.m_model_name = strdup(cpp.m_model_name.c_str());
  return c;
}

c_ChunkingConfig to_c(const ChunkingConfig& cpp)
{
  c_ChunkingConfig c;
  if (std::holds_alternative<FixedSizeChunkingConfig>(cpp.m_config))
  {
    c.m_strategy = FIXED_SIZE_CHUNKING;
    auto& conf = std::get<FixedSizeChunkingConfig>(cpp.m_config);
    c.m_config.m_fixed_size_config.m_chunk_size = conf.m_chunk_size;
    c.m_config.m_fixed_size_config.m_chunk_overlap = conf.m_chunk_overlap;
  }
  return c;
}

c_SemanticSpaceConfig to_c(const SemanticSpaceConfig& cpp)
{
  c_SemanticSpaceConfig c;
  c.m_name = strdup(cpp.m_name.c_str());
  c.m_embedding_model_config = to_c(cpp.m_embedding_model_config);
  c.m_chunking_config = to_c(cpp.m_chunking_config);
  c.m_dimensions = cpp.m_dimensions;
  return c;
}

c_ChatMessage to_c(const ChatMessage& cpp)
{
  c_ChatMessage result;

  // Copy role to fixed-size buffer (truncate if too long)
  strncpy(result.m_role, cpp.m_role.c_str(), sizeof(result.m_role) - 1);
  result.m_role[sizeof(result.m_role) - 1] = '\0';

  // Allocate and copy content
  result.m_content = strdup(cpp.m_content.c_str());

  // Allocate and copy message_metadata as JSON string
  string metadata_json = cpp.m_message_metadata.dump();
  result.m_message_metadata = strdup(metadata_json.c_str());

  result.m_created_at = cpp.m_created_at;

  return result;
}

void to_json(json& j, const ChunkingConfig& p)
{
  if (std::holds_alternative<FixedSizeChunkingConfig>(p.m_config))
  {
    j = json{{"strategy", FIXED_SIZE_CHUNKING}};
    j["config"] = std::get<FixedSizeChunkingConfig>(p.m_config);
  }
}

void from_json(const json& j, ChunkingConfig& p)
{
  ChunkingStrategy strategy;
  j.at("strategy").get_to(strategy);
  if (strategy == FIXED_SIZE_CHUNKING)
  {
    if (j.contains("config"))
    {
      FixedSizeChunkingConfig conf;
      j.at("config").get_to(conf);
      p.m_config = conf;
    }
  }
}