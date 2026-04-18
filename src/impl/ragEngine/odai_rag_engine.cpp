#include "ragEngine/odai_rag_engine.h"
#include "types/odai_types.h"
#include <vector>

#include "backendEngine/odai_backend_engine.h"

#ifdef ODAI_ENABLE_LLAMA_BACKEND
#include "backendEngine/odai_llamacpp/odai_llama_backend_engine.h"
#endif

#ifdef ODAI_ENABLE_SQLITE_DB
#include "db/odai_sqlite/odai_sqlite_db.h"
#endif

#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_type_conversions.h"
#include "utils/odai_helpers.h"

OdaiRagEngine::OdaiRagEngine(const DBConfig& db_config, const BackendEngineConfig& backend_config)
{
  if (db_config.m_dbType == SQLITE_DB)
  {
#ifdef ODAI_ENABLE_SQLITE_DB
    m_db = std::make_unique<OdaiSqliteDb>(db_config);
#else
    throw std::runtime_error("SQLite DB support not enabled");
#endif
  }

  if (backend_config.m_engineType == LLAMA_BACKEND_ENGINE)
  {
#ifdef ODAI_ENABLE_LLAMA_BACKEND
    m_backendEngine = std::make_unique<OdaiLlamaEngine>(backend_config);
#else
    throw std::runtime_error("Llama backend support not enabled");
#endif
  }

  ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully created");
}

OdaiResult<void> OdaiRagEngine::initialize_rag_engine()
{
  if (!m_db)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Database implementation is not available");
    return unexpected_not_initialized();
  }

  OdaiResult<void> db_res = m_db->initialize_db();
  if (!db_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize db, error code: {}", static_cast<std::uint32_t>(db_res.error()));
    return db_res;
  }

  if (!m_backendEngine)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Backend engine implementation is not available");
    return unexpected_not_initialized();
  }

  OdaiResult<void> backend_res = m_backendEngine->initialize_engine();
  if (!backend_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize backend engine, error code: {}",
             static_cast<std::uint32_t>(backend_res.error()));
    return backend_res;
  }

  ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully initialized");
  return {};
}

OdaiResult<void> OdaiRagEngine::register_model_files(const ModelName& name, const ModelFiles& model_file_details)
{
  if (!m_backendEngine || !m_db)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "RAG engine dependencies are not initialized");
    return unexpected_not_initialized();
  }

  OdaiResult<bool> validation_res = m_backendEngine->validate_model_files(model_file_details);
  if (!validation_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "file details validation errored for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(validation_res.error()));
    return tl::unexpected(validation_res.error());
  }

  if (!validation_res.value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "file details validation failed for model: {}", name);
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  OdaiResult<std::string> checksums_res = calculate_model_checksums(model_file_details);
  if (!checksums_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksums for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(checksums_res.error()));
    return tl::unexpected(checksums_res.error());
  }
  const std::string& checksums_json = checksums_res.value();

  OdaiResult<void> db_res = m_db->register_model_files(name, model_file_details, checksums_json);
  if (!db_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to register file details for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(db_res.error()));
    return db_res;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Model files registered successfully for model: {}", name);
  return {};
}

OdaiResult<void> OdaiRagEngine::update_model_files(const ModelName& name, const ModelFiles& new_model_file_details,
                                                   UpdateModelFlag flag)
{
  if (!m_backendEngine || !m_db)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "RAG engine dependencies are not initialized");
    return unexpected_not_initialized();
  }

  OdaiResult<bool> validation_res = m_backendEngine->validate_model_files(new_model_file_details);
  if (!validation_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "new file details validation errored for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(validation_res.error()));
    return tl::unexpected(validation_res.error());
  }

  if (!validation_res.value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "new file details Validation failed for model: {}", name);
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  OdaiResult<std::string> new_checksums_res = calculate_model_checksums(new_model_file_details);
  if (!new_checksums_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksums for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(new_checksums_res.error()));
    return tl::unexpected(new_checksums_res.error());
  }
  const std::string& new_checksums_json = new_checksums_res.value();

  if (flag == UpdateModelFlag::STRICT_MATCH)
  {
    OdaiResult<std::string> old_checksums_res = m_db->get_model_checksums(name);
    if (!old_checksums_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model not found or failed to retrieve checksums: {}, error code: {}", name,
               static_cast<std::uint32_t>(old_checksums_res.error()));
      return tl::unexpected(old_checksums_res.error());
    }
    const std::string& old_checksums_json = old_checksums_res.value();

    nlohmann::json new_checksums_json_obj = nlohmann::json::parse(new_checksums_json);
    nlohmann::json old_checksums_json_obj = nlohmann::json::parse(old_checksums_json);

    for (auto it = new_checksums_json_obj.begin(); it != new_checksums_json_obj.end(); ++it)
    {
      const std::string& key = it.key();
      if (old_checksums_json_obj.contains(key))
      {
        if (old_checksums_json_obj[key] != it.value())
        {
          ODAI_LOG(ODAI_LOG_ERROR, "Checksum mismatch for model: {} on key: {}. Expected: {}, Got: {}", name, key,
                   old_checksums_json_obj[key].dump(), it.value().dump());
          return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
        }
      }
      else
      {
        ODAI_LOG(ODAI_LOG_INFO, "New key found: {}, adding it to the model file details", key);
        old_checksums_json_obj[key] = it.value();
      }
    }
  }

  OdaiResult<void> db_res = m_db->update_model_files(name, new_model_file_details, new_checksums_json);
  if (!db_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to update file details for model: {}, error code: {}", name,
             static_cast<std::uint32_t>(db_res.error()));
    return db_res;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Model files updated successfully for model: {}", name);
  return {};
}

OdaiResult<StreamingStats> OdaiRagEngine::generate_streaming_response(const LLMModelConfig& llm_model_config,
                                                                      const std::vector<InputItem>& prompt,
                                                                      const SamplerConfig& sampler_config,
                                                                      OdaiStreamRespCallbackFn callback,
                                                                      void* user_data)
{
  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  if (prompt.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Query is empty");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  OdaiResult<ModelFiles> model_files_res = resolve_model_files(llm_model_config.m_modelName);
  if (!model_files_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to resolve file details for model: {}", llm_model_config.m_modelName);
    return tl::unexpected(model_files_res.error());
  }
  const ModelFiles& model_files = model_files_res.value();

  const std::vector<InputItem>& processed_prompt = prompt;

  return m_backendEngine->generate_streaming_response(processed_prompt, llm_model_config, model_files, sampler_config,
                                                      callback, user_data);
}

OdaiResult<StreamingStats> OdaiRagEngine::generate_streaming_chat_response(const ChatId& chat_id,
                                                                           const std::vector<InputItem>& prompt,
                                                                           const GeneratorConfig& generator_config,
                                                                           OdaiStreamRespCallbackFn callback,
                                                                           void* user_data)
{
  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  StreamingBufferContext buffer_ctx;
  buffer_ctx.m_userCallback = callback;
  buffer_ctx.m_userData = user_data;
  buffer_ctx.m_bufferedResponse = "";

  // Internal callback that buffers output and forwards to user callback
  auto internal_callback = [](const char* token, void* user_data) -> bool
  {
    if (token == nullptr || user_data == nullptr)
    {
      return false;
    }

    StreamingBufferContext* ctx = static_cast<StreamingBufferContext*>(user_data);

    // Buffer the token
    ctx->m_bufferedResponse += std::string(token);

    // Forward to user callback for streaming
    return ctx->m_userCallback(token, ctx->m_userData);
  };

  // Retrieve chat configuration from database
  OdaiResult<ChatConfig> chat_config_res = m_db->get_chat_config(chat_id);
  if (!chat_config_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to retrieve chat configuration for chat_id: {}", chat_id);
    return tl::unexpected(chat_config_res.error());
  }
  const ChatConfig& chat_config = chat_config_res.value();

  // Check RAG settings: if RAG is enabled but scope_id is empty, return error
  if (generator_config.m_ragMode != RAG_MODE_NEVER)
  {
    if (!generator_config.m_ragConfig.has_value())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled but ragConfig is missing");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    const auto& rag_config = *generator_config.m_ragConfig;

    if (rag_config.m_semanticSpaceName.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled for chat_id: {} but semantic_space_name is empty", chat_id);
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // Retrieve and validate Semantic Space Config
    OdaiResult<SemanticSpaceConfig> space_config_res = m_db->get_semantic_space_config(rag_config.m_semanticSpaceName);
    if (!space_config_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled but failed to retrieve semantic space config for: {}",
               rag_config.m_semanticSpaceName);
      return tl::unexpected(space_config_res.error());
    }

    // ToDo -> Implement context retrieval from knowledge base based on
    // scope_id, using embedding model from spaceConfig Load embedding model
    // from spaceConfig and do similarity check and retrieve according to
    // retrieval strategy string context =
    // retrieve_context_from_knowledge_base(scope_id, prompt, spaceConfig);
    ODAI_LOG(ODAI_LOG_DEBUG, "RAG is enabled for chat_id: {} with space: {} and scope_id: {}", chat_id,
             rag_config.m_semanticSpaceName, rag_config.m_scopeId);
  }

  OdaiResult<std::vector<ChatMessage>> chat_history_res = m_db->get_chat_history(chat_id);
  if (!chat_history_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat history for chat_id: {}", chat_id);
    return tl::unexpected(chat_history_res.error());
  }
  const std::vector<ChatMessage>& chat_history = chat_history_res.value();

  OdaiResult<ModelFiles> model_files_res = resolve_model_files(chat_config.m_llmModelConfig.m_modelName);
  if (!model_files_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to resolve details for model: {}", chat_config.m_llmModelConfig.m_modelName);
    return tl::unexpected(model_files_res.error());
  }
  const ModelFiles& model_files = model_files_res.value();

  const std::vector<InputItem>& prompt_with_context = prompt; // Placeholder until context retrieval is implemented

  std::vector<InputItem> final_prompt = prompt_with_context;

  for (InputItem& item : final_prompt)
  {
    OdaiResult<InputItem> item_res = m_db->store_media_item(item);
    if (!item_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to store media item");
      return tl::unexpected(item_res.error());
    }
    item = item_res.value();
  }

  // Generate streaming response with internal buffering callback
  OdaiResult<StreamingStats> stream_res = m_backendEngine->generate_streaming_chat_response(
      final_prompt, chat_history, chat_config.m_llmModelConfig, model_files, generator_config.m_samplerConfig,
      internal_callback, &buffer_ctx);
  if (!stream_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming response for chat_id: {}, error code: {}", chat_id,
             static_cast<std::uint32_t>(stream_res.error()));
    return tl::unexpected(stream_res.error());
  }

  // Prepare messages to save
  std::vector<ChatMessage> messages_to_save;

  ChatMessage user_msg;
  user_msg.m_role = "user";
  // we pass the modified prompt (here modification means we replace media item with item that we get from
  // store_media_items()) that way we only store and pass file path and not file themselves
  user_msg.m_contentItems = final_prompt;
  // we should update message_metadata with citations if any from RAG context
  user_msg.m_messageMetadata = nlohmann::json::object();
  messages_to_save.push_back(user_msg);

  // ToDo in message_metadata add citations if any from RAG context

  InputItem assistant_item;
  assistant_item.m_type = InputItemType::MEMORY_BUFFER;
  assistant_item.m_data.assign(buffer_ctx.m_bufferedResponse.begin(), buffer_ctx.m_bufferedResponse.end());
  assistant_item.m_mimeType = "text/plain";

  ChatMessage assistant_msg;
  assistant_msg.m_role = "assistant";
  assistant_msg.m_contentItems = {assistant_item};
  assistant_msg.m_messageMetadata = nlohmann::json::object();
  messages_to_save.push_back(assistant_msg);

  // Save messages to database
  OdaiResult<void> save_res = m_db->insert_chat_messages(chat_id, messages_to_save);
  if (!save_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to save messages to database for chat_id: {}", chat_id);
    return tl::unexpected(save_res.error());
  }

  ODAI_LOG(ODAI_LOG_INFO, "Successfully saved chat exchange to database for chat_id: {}", chat_id);

  return stream_res;
}

OdaiResult<ModelFiles> OdaiRagEngine::resolve_model_files(const ModelName& model_name)
{
  OdaiResult<ModelFiles> model_files_res = m_db->get_model_files(model_name);
  if (!model_files_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Model not found in registry: {}", model_name);
    return model_files_res;
  }

  return model_files_res;
}

OdaiResult<void> OdaiRagEngine::create_semantic_space(const SemanticSpaceConfig& config)
{
  return m_db->create_semantic_space(config);
}

OdaiResult<SemanticSpaceConfig> OdaiRagEngine::get_semantic_space_config(const SemanticSpaceName& name)
{
  return m_db->get_semantic_space_config(name);
}

OdaiResult<std::vector<SemanticSpaceConfig>> OdaiRagEngine::list_semantic_spaces()
{
  return m_db->list_semantic_spaces();
}

OdaiResult<void> OdaiRagEngine::delete_semantic_space(const SemanticSpaceName& name)
{
  return m_db->delete_semantic_space(name);
}

OdaiResult<void> OdaiRagEngine::create_chat(const ChatId& chat_id, const ChatConfig& chat_config)
{
  return m_db->create_chat(chat_id, chat_config);
}

OdaiResult<std::vector<ChatMessage>> OdaiRagEngine::get_chat_history(const ChatId& chat_id)
{
  return m_db->get_chat_history(chat_id);
}

OdaiResult<bool> OdaiRagEngine::chat_id_exists(const ChatId& chat_id)
{
  return m_db->chat_id_exists(chat_id);
}
