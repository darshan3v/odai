#include "types/odai_type_conversions.h"
#include "types/odai_ctypes.h"

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

UpdateModelFlag to_cpp_update_model_flag(c_UpdateModelFlag c)
{
  if (c == ODAI_UPDATE_STRICT_MATCH)
  {
    return UpdateModelFlag::STRICT_MATCH;
  }
  return UpdateModelFlag::ALLOW_MISMATCH;
}

InputItemType to_cpp_input_item_type(c_InputItemType c)
{
  switch (c)
  {
  case ODAI_INPUT_ITEM_TYPE_FILE_PATH:
    return InputItemType::FILE_PATH;
  case ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER:
    return InputItemType::MEMORY_BUFFER;
  }

  return InputItemType::FILE_PATH;
}

BackendDeviceType to_cpp_backend_device_type(c_BackendDeviceType c)
{
  switch (c)
  {
  case ODAI_BACKEND_DEVICE_TYPE_CPU:
    return BackendDeviceType::CPU;
  case ODAI_BACKEND_DEVICE_TYPE_GPU:
    return BackendDeviceType::GPU;
  case ODAI_BACKEND_DEVICE_TYPE_IGPU:
    return BackendDeviceType::IGPU;
  case ODAI_BACKEND_DEVICE_TYPE_AUTO:
    return BackendDeviceType::AUTO;
  }
  return BackendDeviceType::AUTO;
}

ModelFiles to_cpp(const c_ModelFiles& c)
{
  ModelFiles model_file_details;
  model_file_details.m_modelType = to_cpp_model_type(c.m_modelType);
  model_file_details.m_engineType = c.m_engineType;

  if ((c.m_entries != nullptr) && c.m_entriesCount > 0)
  {
    for (size_t i = 0; i < c.m_entriesCount; ++i)
    {
      if ((c.m_entries[i].m_key != nullptr) && (c.m_entries[i].m_value != nullptr))
      {
        model_file_details.m_entries[std::string(c.m_entries[i].m_key)] = std::string(c.m_entries[i].m_value);
      }
    }
  }
  return model_file_details;
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
  return {c.m_dbType, std::string(c.m_dbPath), std::string(c.m_mediaStorePath)};
}

BackendEngineConfig to_cpp(const c_BackendEngineConfig& c)
{
  BackendEngineConfig cpp_config{};
  cpp_config.m_engineType = c.m_engineType;
  cpp_config.m_preferredDeviceType = to_cpp_backend_device_type(c.m_preferredDeviceType);
  return cpp_config;
}

EmbeddingModelConfig to_cpp(const c_EmbeddingModelConfig& c)
{
  return {std::string(c.m_modelName)};
}

LLMModelConfig to_cpp(const c_LlmModelConfig& c)
{
  return {std::string(c.m_modelName)};
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
  config.m_name = std::string(c.m_name);
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
    config.m_semanticSpaceName = std::string(source.m_semanticSpaceName);
  }
  if (source.m_scopeId != nullptr)
  {
    config.m_scopeId = std::string(source.m_scopeId);
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
  return {c.m_persistence, std::string(c.m_systemPrompt), to_cpp(c.m_llmModelConfig)};
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
  case InputItemType::FILE_PATH:
    return ODAI_INPUT_ITEM_TYPE_FILE_PATH;
  case InputItemType::MEMORY_BUFFER:
    return ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER;
  default:
    return ODAI_INPUT_ITEM_TYPE_INVALID;
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
  std::string metadata_json = cpp.m_messageMetadata.dump();
  result.m_messageMetadata = strdup(metadata_json.c_str());

  result.m_createdAt = cpp.m_createdAt;

  return result;
}

std::string byte_vector_to_string(const std::vector<uint8_t>& bytes)
{
  return std::string(bytes.begin(), bytes.end());
}

void to_json(nlohmann::json& j, const ChunkingConfig& p)
{
  if (std::holds_alternative<FixedSizeChunkingConfig>(p.m_config))
  {
    j = nlohmann::json{{"strategy", FIXED_SIZE_CHUNKING}};
    j["config"] = std::get<FixedSizeChunkingConfig>(p.m_config);
  }
}

void from_json(const nlohmann::json& j, ChunkingConfig& p)
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
