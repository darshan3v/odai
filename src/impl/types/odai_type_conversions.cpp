#include "types/odai_type_conversions.h"

#include <cstring>

ModelType to_cpp_model_type(c_ModelType c)
{
  if (c == ODAI_MODEL_TYPE_LLM)
  {
    return ModelType::LLM;
  }
  if (c == ODAI_MODEL_TYPE_EMBEDDING)
  {
    return ModelType::EMBEDDING;
  }

  // Default to LLM if unknown, or handle error appropriately.
  // Since we can't easily return error here, we assume valid input or handle at caller.
  return ModelType::LLM;
}

InputItemType to_cpp_input_item_type(c_InputItemType c)
{
  switch (c)
  {
  case ODAI_INPUT_ITEM_TYPE_TEXT:
    return InputItemType::TEXT;
  case ODAI_INPUT_ITEM_TYPE_IMAGE_FILE:
    return InputItemType::IMAGE_FILE;
  case ODAI_INPUT_ITEM_TYPE_AUDIO_FILE:
    return InputItemType::AUDIO_FILE;
  case ODAI_INPUT_ITEM_TYPE_IMAGE_BASE64:
    return InputItemType::IMAGE_BASE64;
  case ODAI_INPUT_ITEM_TYPE_AUDIO_BASE64:
    return InputItemType::AUDIO_BASE64;
  default:
    return InputItemType::TEXT;
  }
}

InputItem to_cpp(const c_InputItem& c)
{
  InputItem item;
  item.m_type = to_cpp_input_item_type(c.m_type);

  if ((c.m_data != nullptr) && c.m_dataSize > 0)
  {
    const uint8_t* ptr = static_cast<const uint8_t*>(c.m_data);
    item.m_data.assign(ptr, ptr + c.m_dataSize);
  }

  if (c.m_mimeType != nullptr)
  {
    item.m_mimeType = c.m_mimeType;
  }

  return item;
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
  RetrievalConfig config{};
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
  if (source.m_semanticSpaceName != nullptr)
  {
    config.m_semanticSpaceName = string(source.m_semanticSpaceName);
  }
  if (source.m_scopeId != nullptr)
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
  c_EmbeddingModelConfig c{};
  c.m_modelName = strdup(cpp.m_modelName.c_str());
  return c;
}

c_InputItemType to_c(const InputItemType& cpp)
{
  switch (cpp)
  {
  case InputItemType::TEXT:
    return ODAI_INPUT_ITEM_TYPE_TEXT;
  case InputItemType::IMAGE_FILE:
    return ODAI_INPUT_ITEM_TYPE_IMAGE_FILE;
  case InputItemType::AUDIO_FILE:
    return ODAI_INPUT_ITEM_TYPE_AUDIO_FILE;
  case InputItemType::IMAGE_BASE64:
    return ODAI_INPUT_ITEM_TYPE_IMAGE_BASE64;
  case InputItemType::AUDIO_BASE64:
    return ODAI_INPUT_ITEM_TYPE_AUDIO_BASE64;
  default:
    return ODAI_INPUT_ITEM_TYPE_TEXT;
  }
}

c_InputItem to_c(const InputItem& cpp)
{
  c_InputItem item{};
  item.m_type = to_c(cpp.m_type);
  item.m_dataSize = cpp.m_data.size();

  if (item.m_dataSize > 0)
  {
    item.m_data = malloc(item.m_dataSize);
    memcpy(item.m_data, cpp.m_data.data(), item.m_dataSize);
  }
  else
  {
    item.m_data = nullptr;
  }

  if (!cpp.m_mimeType.empty())
  {
    item.m_mimeType = strdup(cpp.m_mimeType.c_str());
  }
  else
  {
    item.m_mimeType = nullptr;
  }

  return item;
}

c_ChunkingConfig to_c(const ChunkingConfig& cpp)
{
  c_ChunkingConfig c{};
  if (std::holds_alternative<FixedSizeChunkingConfig>(cpp.m_config))
  {
    c.m_strategy = FIXED_SIZE_CHUNKING;
    const auto& conf = std::get<FixedSizeChunkingConfig>(cpp.m_config);
    c.m_config.m_fixedSizeConfig.m_chunkSize = conf.m_chunkSize;
    c.m_config.m_fixedSizeConfig.m_chunkOverlap = conf.m_chunkOverlap;
  }
  return c;
}

c_SemanticSpaceConfig to_c(const SemanticSpaceConfig& cpp)
{
  c_SemanticSpaceConfig c{};
  c.m_name = strdup(cpp.m_name.c_str());
  c.m_embeddingModelConfig = to_c(cpp.m_embeddingModelConfig);
  c.m_chunkingConfig = to_c(cpp.m_chunkingConfig);
  c.m_dimensions = cpp.m_dimensions;
  return c;
}

c_ChatMessage to_c(const ChatMessage& cpp)
{
  c_ChatMessage result{};

  // Copy role to fixed-size buffer (truncate if too long)
  strncpy(result.m_role, cpp.m_role.c_str(), sizeof(result.m_role) - 1);
  result.m_role[sizeof(result.m_role) - 1] = '\0';

  // Allocate and copy content items
  result.m_contentItemsCount = cpp.m_contentItems.size();
  if (result.m_contentItemsCount > 0)
  {
    result.m_contentItems = static_cast<c_InputItem*>(malloc(sizeof(c_InputItem) * result.m_contentItemsCount));
    for (size_t i = 0; i < result.m_contentItemsCount; ++i)
    {
      result.m_contentItems[i] = to_c(cpp.m_contentItems[i]);
    }
  }
  else
  {
    result.m_contentItems = nullptr;
  }

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
  ChunkingStrategy strategy = 0;
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