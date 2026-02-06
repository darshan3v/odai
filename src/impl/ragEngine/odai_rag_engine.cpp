#include "ragEngine/odai_rag_engine.h"
#include "backendEngine/odai_backend_engine.h"
#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "utils/odai_helpers.h"

ODAIRagEngine::ODAIRagEngine(ODAIDb* db, ODAIBackendEngine* backend_engine)
{
  this->m_db = db;
  this->m_backendEngine = backend_engine;

  ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully initialized");
}

bool ODAIRagEngine::register_model(const ModelName& name, const ModelPath& path, ModelType type)
{
  string checksum = calculate_file_checksum(path);
  if (checksum.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksum for file: {}", path);
    return false;
  }

  if (m_db->register_model(name, path, type, checksum))
  {
    // Update cache
    m_modelPathCache[name] = path;
    return true;
  }
  return false;
}

bool ODAIRagEngine::update_model_path(const ModelName& name, const ModelPath& path)
{
  const string CHECKSUM = calculate_file_checksum(path);
  if (CHECKSUM.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksum for file: {}", path);
    return false;
  }

  string old_checksum;
  if (!m_db->get_model_checksum(name, old_checksum))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Model not found or failed to retrieve checksum: {}", name);
    return false;
  }

  if (CHECKSUM != old_checksum)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Checksum mismatch for model: {}. Expected: {}, Got: {}", name, old_checksum, CHECKSUM);
    return false;
  }

  if (m_db->update_model_path(name, path))
  {
    // Update cache
    m_modelPathCache[name] = path;
    return true;
  }
  return false;
}

int32_t ODAIRagEngine::generate_streaming_response(const LLMModelConfig& llm_model_config, const string& query,
                                                   const SamplerConfig& sampler_config,
                                                   OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return -1;
  }

  if (query.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Query is empty");
    return -1;
  }

  ModelPath model_path;
  if (!this->resolve_model_path(llm_model_config.m_modelName, model_path))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to resolve path for model: {}", llm_model_config.m_modelName);
    return -1;
  }

  if (!m_backendEngine->load_language_model(model_path, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load given language model");
    return -1;
  }

  return m_backendEngine->generate_streaming_response(query, sampler_config, callback, user_data);
}

bool ODAIRagEngine::load_chat_session(const ChatId& chat_id)
{
  ChatConfig chat_config;

  if (!m_db->get_chat_config(chat_id, chat_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat config, chat_id: {}", chat_id);
    return false;
  }

  return this->ensure_chat_session_loaded(chat_id, chat_config);
}

int32_t ODAIRagEngine::generate_streaming_chat_response(const ChatId& chat_id, const string& prompt,
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

  if (!this->ensure_chat_session_loaded(chat_id, chat_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to ensure chat session is loaded for chat_id: {}", chat_id);
    return -1;
  }

  // final_prompt = combine_context_and_query(context, prompt);
  string final_prompt = prompt; // Placeholder until context retrieval is implemented

  StreamingBufferContext buffer_ctx;
  buffer_ctx.m_userCallback = callback;
  buffer_ctx.m_userData = user_data;
  buffer_ctx.m_bufferedResponse = "";

  // Internal callback that buffers output and forwards to user callback
  auto internal_callback = [](const char* token, void* user_data) -> bool
  {
    if (token == nullptr || user_data == nullptr)
      return false;

    StreamingBufferContext* ctx = static_cast<StreamingBufferContext*>(user_data);

    // Buffer the token
    ctx->m_bufferedResponse += string(token);

    // Forward to user callback for streaming
    return ctx->m_userCallback(token, ctx->m_userData);
  };

  // Generate streaming response with internal buffering callback
  int32_t total_tokens = m_backendEngine->generate_streaming_chat_response(
      chat_id, final_prompt, generator_config.m_samplerConfig, internal_callback, &buffer_ctx);

  if (total_tokens < 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming response for chat_id: {}", chat_id);
    return -1;
  }

  // Prepare messages to save
  vector<ChatMessage> messages_to_save;

  ChatMessage user_msg;
  user_msg.m_role = "user";
  user_msg.m_content = prompt;
  user_msg.m_messageMetadata = json::object();
  messages_to_save.push_back(user_msg);

  // ToDo in message_metadata add citations if any from RAG context

  ChatMessage assistant_msg;
  assistant_msg.m_role = "assistant";
  assistant_msg.m_content = buffer_ctx.m_bufferedResponse;
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

bool ODAIRagEngine::unload_chat_session(const ChatId& chat_id)
{
  if (this->m_backendEngine == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Backend engine is null");
    return false;
  }
  return this->m_backendEngine->unload_chat_context(chat_id);
}

bool ODAIRagEngine::resolve_model_path(const ModelName& model_name, ModelPath& path)
{
  // check cache first
  auto it = m_modelPathCache.find(model_name);
  if (it != m_modelPathCache.end())
  {
    path = it->second;
    return true;
  }

  // if not in cache, check db
  if (m_db->get_model_path(model_name, path))
  {
    // update cache
    m_modelPathCache[model_name] = path;
    return true;
  }

  ODAI_LOG(ODAI_LOG_ERROR, "Model not found in registry: {}", model_name);
  return false;
}

bool ODAIRagEngine::ensure_chat_session_loaded(const ChatId& chat_id, const ChatConfig& chat_config)
{
  // 1. Ensure the correct language model is loaded
  // This call handles checking if the model is already loaded (fast path),
  // and if not, it loads it and CLEARS existing contexts (slow path).
  ModelPath model_path;
  if (!resolve_model_path(chat_config.m_llmModelConfig.m_modelName, model_path))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to resolve model path for chat {}", chat_id);
    return false;
  }

  if (!m_backendEngine->load_language_model(model_path, chat_config.m_llmModelConfig))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat {}, error : failed to load language Model", chat_id);
    return false;
  }

  // 2. Check if context is already loaded
  if (m_backendEngine->is_chat_context_loaded(chat_id))
  {
    // Context is loaded, we are good to go
    return true;
  }

  vector<ChatMessage> messages;
  if (!m_db->get_chat_history(chat_id, messages))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat history for chat_id: {}", chat_id);
    return false;
  }

  if (!m_backendEngine->load_chat_messages_into_context(chat_id, messages))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat history into context for chat_id: {}", chat_id);
    return false;
  }

  return true;
}