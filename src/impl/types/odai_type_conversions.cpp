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
  return {c.m_dbType, string(c.m_dbPath)};
}

BackendEngineConfig to_cpp(const c_BackendEngineConfig& c)
{
  return {c.m_engineType};
}

EmbeddingModelConfig to_cpp(const c_EmbeddingModelConfig& c)
{
  return {string(c.m_modelName)};
}

LLMModelConfig to_cpp(const c_LlmModelConfig& c)
{
  return {string(c.m_modelName)};
}

ChunkingConfig to_cpp(const c_ChunkingConfig& c)
{
  ChunkingConfig config;

  if (c.m_strategy == FIXED_SIZE_CHUNKING)
  {
    FixedSizeChunkingConfig fcc;
    fcc.m_chunkSize = c.m_config.m_fixedSizeConfig.m_chunkSize;
    fcc.m_chunkOverlap = c.m_config.m_fixedSizeConfig.m_chunkOverlap;
    config.m_config = fcc;
  }
  return config;
}

SemanticSpaceConfig to_cpp(const c_SemanticSpaceConfig& c)
{
  SemanticSpaceConfig config;
  config.m_name = string(c.m_name);
  config.m_embeddingModelConfig = to_cpp(c.m_embeddingModelConfig);
  config.m_chunkingConfig = to_cpp(c.m_chunkingConfig);
  config.m_dimensions = c.m_dimensions;
  return config;
}

RetrievalConfig to_cpp(const c_RetrievalConfig& c)
{
  RetrievalConfig config;
  config.m_topK = c.m_topK;
  config.m_fetchK = c.m_fetchK;
  config.m_scoreThreshold = c.m_scoreThreshold;
  config.m_searchType = c.m_searchType;
  config.m_useReranker = c.m_useReranker;
  config.m_contextWindow = c.m_contextWindow;
  return config;
}

SamplerConfig to_cpp(const c_SamplerConfig& c)
{
  return {c.m_maxTokens, c.m_topP, c.m_topK};
}

GeneratorRagConfig to_cpp(const c_GeneratorRagConfig& source)
{
  GeneratorRagConfig config;
  config.m_retrievalConfig = to_cpp(source.m_retrievalConfig);
  if (source.m_semanticSpaceName)
  {
    config.m_semanticSpaceName = string(source.m_semanticSpaceName);
  }
  if (source.m_scopeId)
  {
    config.m_scopeId = string(source.m_scopeId);
  }
  return config;
}

GeneratorConfig to_cpp(const c_GeneratorConfig& source)
{
  GeneratorConfig config;
  config.m_samplerConfig = to_cpp(source.m_samplerConfig);
  config.m_ragMode = source.m_ragMode;

  if (source.m_ragConfig != nullptr)
  {
    config.m_ragConfig = to_cpp(*source.m_ragConfig);
  }
  else
  {
    config.m_ragConfig = std::nullopt;
  }

  return config;
}

ChatConfig to_cpp(const c_ChatConfig& c)
{
  return {c.m_persistence, string(c.m_systemPrompt), to_cpp(c.m_llmModelConfig)};
}

c_EmbeddingModelConfig to_c(const EmbeddingModelConfig& cpp)
{
  c_EmbeddingModelConfig c;
  c.m_modelName = strdup(cpp.m_modelName.c_str());
  return c;
}

c_ChunkingConfig to_c(const ChunkingConfig& cpp)
{
  c_ChunkingConfig c;
  if (std::holds_alternative<FixedSizeChunkingConfig>(cpp.m_config))
  {
    c.m_strategy = FIXED_SIZE_CHUNKING;
    auto& conf = std::get<FixedSizeChunkingConfig>(cpp.m_config);
    c.m_config.m_fixedSizeConfig.m_chunkSize = conf.m_chunkSize;
    c.m_config.m_fixedSizeConfig.m_chunkOverlap = conf.m_chunkOverlap;
  }
  return c;
}

c_SemanticSpaceConfig to_c(const SemanticSpaceConfig& cpp)
{
  c_SemanticSpaceConfig c;
  c.m_name = strdup(cpp.m_name.c_str());
  c.m_embeddingModelConfig = to_c(cpp.m_embeddingModelConfig);
  c.m_chunkingConfig = to_c(cpp.m_chunkingConfig);
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
  string metadata_json = cpp.m_messageMetadata.dump();
  result.m_messageMetadata = strdup(metadata_json.c_str());

  result.m_createdAt = cpp.m_createdAt;

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