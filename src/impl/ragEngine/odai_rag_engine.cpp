#include "ragEngine/odai_rag_engine.h"
#include "types/odai_types.h"
#include <vector>

#ifdef ODAI_ENABLE_MINIAUDIO
#include "audioEngine/odai_miniaudio_decoder.h"
#endif

#include "backendEngine/odai_backend_engine.h"

#ifdef ODAI_ENABLE_LLAMA_BACKEND
#include "backendEngine/odai_llama_backend_engine.h"
#endif

#ifdef ODAI_ENABLE_SQLITE_DB
#include "db/odai_sqlite_db.h"
#endif

#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_type_conversions.h"
#include "utils/odai_helpers.h"
#include <cstring>
#include <stdexcept>

OdaiRagEngine::OdaiRagEngine(const DBConfig& db_config, const BackendEngineConfig& backend_config,
                             const SdkConfig& sdk_config)
{
  if (db_config.m_dbType == SQLITE_DB)
  {
#ifdef ODAI_ENABLE_SQLITE_DB
    m_db = std::make_unique<OdaiSqliteDb>(db_config, sdk_config.m_cacheDirPath);
#else
    throw std::runtime_error("SQLite DB support not enabled");
#endif
  }

  if (!m_db || !m_db->initialize_db())
  {
    throw std::runtime_error("Failed to initialize db in RAG engine");
  }

#ifdef ODAI_ENABLE_MINIAUDIO
  m_audioDecoder = std::make_unique<OdaiMiniAudioDecoder>();
  ODAI_LOG(ODAI_LOG_INFO, "OdaiMiniAudioDecoder initialized in RAG engine");
#else
  throw std::runtime_error("MiniAudio support not enabled");
#endif

  if (backend_config.m_engineType == LLAMA_BACKEND_ENGINE)
  {
#ifdef ODAI_ENABLE_LLAMA_BACKEND
    m_backendEngine = std::make_unique<OdaiLlamaEngine>(backend_config);
#else
    throw std::runtime_error("Llama backend support not enabled");
#endif
  }

  if (!m_backendEngine || !m_backendEngine->initialize_engine())
  {
    throw std::runtime_error("Failed to initialize backend engine in RAG engine");
  }

  ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully initialized");
}

bool OdaiRagEngine::register_model_files(const ModelName& name, const ModelFiles& details)
{
  if (!m_backendEngine->validate_model_files(details))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Validation failed for model details: {}", name);
    return false;
  }

  string checksums_json = calculate_model_checksums(details);
  if (checksums_json.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksums for model: {}", name);
    return false;
  }

  if (m_db->register_model_files(name, details, checksums_json))
  {
    // Update cache
    m_modelDetailsCache[name] = details;
    return true;
  }
  return false;
}

bool OdaiRagEngine::update_model_files(const ModelName& name, const ModelFiles& new_details, UpdateModelFlag flag)
{
  if (!m_backendEngine->validate_model_files(new_details))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Validation failed for model details: {}", name);
    return false;
  }

  string new_checksums_json = calculate_model_checksums(new_details);
  if (new_checksums_json.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksums for model: {}", name);
    return false;
  }

  if (flag == UpdateModelFlag::STRICT_MATCH)
  {
    string old_checksums_json;
    if (!m_db->get_model_checksums(name, old_checksums_json))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model not found or failed to retrieve checksums: {}", name);
      return false;
    }

    try
    {
      auto new_checksums_json_obj = nlohmann::json::parse(new_checksums_json);
      auto old_checksums_json_obj = nlohmann::json::parse(old_checksums_json);

      for (auto it = new_checksums_json_obj.begin(); it != new_checksums_json_obj.end(); ++it)
      {
        const string& key = it.key();
        if (old_checksums_json_obj.contains(key))
        {
          if (old_checksums_json_obj[key] != it.value())
          {
            ODAI_LOG(ODAI_LOG_ERROR, "Checksum mismatch for model: {} on key: {}. Expected: {}, Got: {}", name, key,
                     old_checksums_json_obj[key].dump(), it.value().dump());
            return false;
          }
        }
        else
        {
          ODAI_LOG(ODAI_LOG_INFO, "New key found: {}, adding it to the model details", key);
          old_checksums_json_obj[key] = it.value();
        }
      }
    }
    catch (...)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to parse/compare checksum JSONs");
      return false;
    }
  }

  if (m_db->update_model_files(name, new_details, new_checksums_json))
  {
    // Update cache
    m_modelDetailsCache[name] = new_details;
    return true;
  }
  return false;
}

int32_t OdaiRagEngine::generate_streaming_response(const LLMModelConfig& llm_model_config,
                                                   const vector<InputItem>& prompt, const SamplerConfig& sampler_config,
                                                   OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return -1;
  }

  if (prompt.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Query is empty");
    return -1;
  }

  ModelFiles model_files;
  if (!resolve_model_files(llm_model_config.m_modelName, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to resolve details for model: {}", llm_model_config.m_modelName);
    return -1;
  }

  vector<InputItem> processed_prompt = prompt;
  if (!this->process_multimodal_inputs(processed_prompt, llm_model_config, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load multimodal inputs");
    return -1;
  }

  return m_backendEngine->generate_streaming_response(processed_prompt, llm_model_config, model_files, sampler_config,
                                                      callback, user_data);
}

int32_t OdaiRagEngine::generate_streaming_chat_response(const ChatId& chat_id, const vector<InputItem>& prompt,
                                                        const GeneratorConfig& generator_config,
                                                        OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return -1;
  }

  // Retrieve chat configuration from database
  ChatConfig chat_config;
  if (!m_db->get_chat_config(chat_id, chat_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to retrieve chat configuration for chat_id: {}", chat_id);
    return -1;
  }

  // Check RAG settings: if RAG is enabled but scope_id is empty, return error
  if (generator_config.m_ragMode != RAG_MODE_NEVER)
  {
    if (!generator_config.m_ragConfig.has_value())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled but ragConfig is missing");
      return -1;
    }

    const auto& rag_config = *generator_config.m_ragConfig;

    if (rag_config.m_semanticSpaceName.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled for chat_id: {} but semantic_space_name is empty", chat_id);
      return -1;
    }

    // Retrieve and validate Semantic Space Config
    SemanticSpaceConfig space_config;
    if (!m_db->get_semantic_space_config(rag_config.m_semanticSpaceName, space_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled but failed to retrieve semantic space config for: {}",
               rag_config.m_semanticSpaceName);
      return -1;
    }

    // ToDo -> Implement context retrieval from knowledge base based on
    // scope_id, using embedding model from spaceConfig Load embedding model
    // from spaceConfig and do similarity check and retrieve according to
    // retrieval strategy string context =
    // retrieve_context_from_knowledge_base(scope_id, prompt, spaceConfig);
    ODAI_LOG(ODAI_LOG_DEBUG, "RAG is enabled for chat_id: {} with space: {} and scope_id: {}", chat_id,
             rag_config.m_semanticSpaceName, rag_config.m_scopeId);
  }

  vector<ChatMessage> chat_history;
  if (!m_db->get_chat_history(chat_id, chat_history))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat history for chat_id: {}", chat_id);
    return -1;
  }

  ModelFiles model_files;
  if (!resolve_model_files(chat_config.m_llmModelConfig.m_modelName, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to resolve details for model: {}", chat_config.m_llmModelConfig.m_modelName);
    return -1;
  }

  for (auto& msg : chat_history)
  {
    if (!this->process_multimodal_inputs(msg.m_contentItems, chat_config.m_llmModelConfig, model_files))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat history multimodal inputs");
      return -1;
    }
  }

  const vector<InputItem>& final_prompt = prompt; // Placeholder until context retrieval is implemented

  vector<InputItem> processed_final_prompt = final_prompt;

  if (!this->process_multimodal_inputs(processed_final_prompt, chat_config.m_llmModelConfig, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to process multimodal inputs");
    return -1;
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
    ctx->m_bufferedResponse += string(token);

    // Forward to user callback for streaming
    return ctx->m_userCallback(token, ctx->m_userData);
  };

  // Generate streaming response with internal buffering callback
  int32_t total_tokens = m_backendEngine->generate_streaming_chat_response(
      chat_id, chat_history, processed_final_prompt, generator_config.m_samplerConfig, internal_callback, &buffer_ctx);

  if (total_tokens < 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming response for chat_id: {}", chat_id);
    return -1;
  }

  // Prepare messages to save
  vector<ChatMessage> messages_to_save;

  ChatMessage user_msg;
  user_msg.m_role = "user";
  // We MUST pass the original 'prompt' to the DB so that we serialize
  // the true reference/buffer (e.g., AUDIO_FILE) and not the internal decoded data
  user_msg.m_contentItems = final_prompt;
  // we should update message_metadata with citations if any from RAG context
  user_msg.m_messageMetadata = json::object();
  messages_to_save.push_back(user_msg);

  // ToDo in message_metadata add citations if any from RAG context

  InputItem assistant_item;
  assistant_item.m_type = InputItemType::MEMORY_BUFFER;
  assistant_item.m_data.assign(buffer_ctx.m_bufferedResponse.begin(), buffer_ctx.m_bufferedResponse.end());
  assistant_item.m_mimeType = "text/plain";

  ChatMessage assistant_msg;
  assistant_msg.m_role = "assistant";
  assistant_msg.m_contentItems = {assistant_item};
  assistant_msg.m_messageMetadata = json::object();
  messages_to_save.push_back(assistant_msg);

  if (!m_db->begin_transaction())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to begin transaction for chat_id: {}", chat_id);
    return -1;
  }

  // Save messages to database
  if (!m_db->insert_chat_messages(chat_id, messages_to_save))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to save messages to database for chat_id: {}", chat_id);
    m_db->rollback_transaction();
    return -1;
  }

  if (!m_db->commit_transaction())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction for chat_id: {}", chat_id);
    m_db->rollback_transaction();
    return -1;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Successfully saved chat exchange to database for chat_id: {}", chat_id);

  return total_tokens;
}

bool OdaiRagEngine::resolve_model_files(const ModelName& model_name, ModelFiles& details)
{
  // check cache first
  auto it = m_modelDetailsCache.find(model_name);
  if (it != m_modelDetailsCache.end())
  {
    details = it->second;
    return true;
  }

  // if not in cache, check db
  if (m_db->get_model_files(model_name, details))
  {
    // update cache
    m_modelDetailsCache[model_name] = details;
    return true;
  }

  ODAI_LOG(ODAI_LOG_ERROR, "Model not found in registry: {}", model_name);
  return false;
}

bool OdaiRagEngine::process_multimodal_inputs(vector<InputItem>& prompt_out, const LLMModelConfig& llm_model_config,
                                              const ModelFiles& model_files)
{
  std::optional<OdaiAudioTargetSpec> audio_spec_opt =
      m_backendEngine->get_required_audio_spec(llm_model_config, model_files);

  for (auto& item : prompt_out)
  {

    if (item.m_type == InputItemType::PROCESSED_DATA)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Undefined, we expect unprocessed data in this function");
      return false;
    }

    MediaType media_type = get_media_type_from_mime(item.m_mimeType);

    if (media_type == MediaType::TEXT)
    {
      // no special processing needed for text
      item.m_type = InputItemType::PROCESSED_DATA;
    }
    else if (media_type == MediaType::AUDIO)
    {
      if (audio_spec_opt.has_value())
      {
        if (!m_audioDecoder)
        {
          ODAI_LOG(ODAI_LOG_ERROR, "Target model expects audio but OdaiAudioDecoder is not available");
          return false;
        }

        OdaiDecodedAudio decoded_audio;
        bool decode_success = m_audioDecoder->decode_to_spec(item, audio_spec_opt.value(), decoded_audio);

        if (!decode_success)
        {
          ODAI_LOG(ODAI_LOG_ERROR, "Failed to decode input audio item");
          return false;
        }

        // Convert float array to uint8_t byte array for transport
        size_t byte_size = decoded_audio.m_samples.size() * sizeof(float);
        std::vector<uint8_t> pcm_bytes(byte_size);
        std::memcpy(pcm_bytes.data(), decoded_audio.m_samples.data(), byte_size);

        item.m_data = std::move(pcm_bytes);
        item.m_type = InputItemType::PROCESSED_DATA;
      }
      else
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Input contains audio but backend model does not support it.");
        return false;
      }
    }
  }

  return true;
}

bool OdaiRagEngine::create_semantic_space(const SemanticSpaceConfig& config)
{
  return m_db->create_semantic_space(config);
}

bool OdaiRagEngine::get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config)
{
  return m_db->get_semantic_space_config(name, config);
}

bool OdaiRagEngine::list_semantic_spaces(vector<SemanticSpaceConfig>& spaces)
{
  return m_db->list_semantic_spaces(spaces);
}

bool OdaiRagEngine::delete_semantic_space(const SemanticSpaceName& name)
{
  return m_db->delete_semantic_space(name);
}

bool OdaiRagEngine::create_chat(const ChatId& chat_id, const ChatConfig& chat_config)
{
  return m_db->create_chat(chat_id, chat_config);
}

bool OdaiRagEngine::get_chat_history(const ChatId& chat_id, vector<ChatMessage>& messages)
{
  return m_db->get_chat_history(chat_id, messages);
}

bool OdaiRagEngine::chat_id_exists(const ChatId& chat_id)
{
  return m_db->chat_id_exists(chat_id);
}