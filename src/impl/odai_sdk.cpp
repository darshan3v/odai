#include "odai_sdk.h"
#include "ragEngine/odai_rag_engine.h"

#include "backendEngine/odai_llama_backend_engine.h"
#include "db/odai_sqlite_db.h"
#include "types/odai_common_types.h"
#include "types/odai_types.h"
#include "utils/odai_helpers.h"

#include <memory>

using namespace std;

ODAISdk& ODAISdk::get_instance()
{
  static ODAISdk instance;
  return instance;
}

ODAISdk::ODAISdk()
{
  m_logger = std::make_unique<ODAILogger>();
}

ODAISdk::~ODAISdk()
{
  // Clean up if needed, though smart pointers in globals handle themselves
  // mostly But we might want to shut down explicitly if order matters
}

void ODAISdk::set_logger(OdaiLogCallbackFn callback, void* user_data)
{
  try
  {
    m_logger->set_logger(callback, user_data);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
  }
}

void ODAISdk::set_log_level(OdaiLogLevel log_level)
{
  try
  {
    m_logger->set_log_level(log_level);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
  }
}

bool ODAISdk::initialize_sdk(const DBConfig& db_config, const BackendEngineConfig& backend_config)
{
  try
  {
    if (!db_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid db config passed");
      return false;
    }

    if (db_config.m_db_type == SQLITE_DB)
    {
      m_db = std::make_unique<ODAISqliteDb>(db_config);
    }

    if ((m_db.get() == nullptr) || (!m_db->initialize_db()))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize db");
      m_sdk_initialized = false;
      return false;
    }

    if (!backend_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid backend engine config passed");
      return false;
    }

    // Initialize the backend engine based on engineType
    if (backend_config.m_engine_type == LLAMA_BACKEND_ENGINE)
      m_backend_engine = std::make_unique<ODAILlamaEngine>(backend_config);

    if ((m_backend_engine.get() == nullptr) || (!m_backend_engine->initialize_engine()))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize backend engine");
      m_sdk_initialized = false;
      return false;
    }

    // Initalize the RAGEngine
    m_rag_engine = make_unique<ODAIRagEngine>(m_db.get(), m_backend_engine.get());

    if ((m_rag_engine.get() == nullptr))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize RAG engine");
      m_sdk_initialized = false;
      return false;
    }

    m_sdk_initialized = true;
    ODAI_LOG(ODAI_LOG_INFO, "ODAI SDK Initialized successfully");
    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    m_sdk_initialized = false;
    return false;
  }
}

bool ODAISdk::register_model(const ModelName& name, const ModelPath& path, ModelType type)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_rag_engine->register_model(name, path, type);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::update_model_path(const ModelName& name, const ModelPath& path)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_rag_engine->update_model_path(name, path);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::create_semantic_space(const SemanticSpaceConfig& config)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (config.is_sane() == false)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid semantic space config passed");
      return false;
    }

    // TODO: if dim == 0 then auto infer from model

    return m_db->create_semantic_space(config);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_db->get_semantic_space_config(name, config);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::list_semantic_spaces(vector<SemanticSpaceConfig>& spaces)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_db->list_semantic_spaces(spaces);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::delete_semantic_space(const SemanticSpaceName& name)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_db->delete_semantic_space(name);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::add_document(const string& content, const DocumentId& document_id,
                           const SemanticSpaceName& semantic_space_name, const ScopeId& scope_id)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    // ToDo : call rag engine to add document

    ODAI_LOG(ODAI_LOG_INFO, "Adding document to space: {}", semantic_space_name);

    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

int32_t ODAISdk::generate_streaming_response(const LLMModelConfig& llm_model_config, const string& query,
                                             const SamplerConfig& sampler_config, OdaiStreamRespCallbackFn callback,
                                             void* user_data)
{
  try
  {

    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return -1;
    }

    if (!llm_model_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid LLM Model Config passed");
      return -1;
    }

    if (!sampler_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid Sampler Config passed");
      return -1;
    }

    if (query.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
      return -1;
    }

    if (callback == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "empty callback passed");
      return -1;
    }

    int32_t total_tokens =
        m_rag_engine->generate_streaming_response(llm_model_config, query, sampler_config, callback, user_data);
    if (total_tokens < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to generate response");
      return -1;
    }

    return total_tokens;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return -1;
  }
}

bool ODAISdk::create_chat(const ChatId& chat_id_in, const ChatConfig& chat_config, ChatId& chat_id_out)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    ChatId chat_id;

    if (!chat_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
      return false;
    }

    // if chat_id is empty then we will generate and set the out, else we will
    // use the given chat_id, given it's unique
    if (chat_id_in.empty())
    {
      chat_id = generate_chat_id();
      chat_id_out = chat_id;
    }
    else
    {
      chat_id = chat_id_in;
      chat_id_out = chat_id; // Set output to input in this case
      if (m_db->chat_id_exists(chat_id))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} already exists", chat_id_in);
        return false;
      }
    }

    if (!m_db->create_chat(chat_id, chat_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to create chat");
      return false;
    }

    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::load_chat(const ChatId& chat_id)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return false;
    }

    if (!m_rag_engine->load_chat_session(chat_id))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat session, chat_id: {}", chat_id);
      return false;
    }

    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::get_chat_history(const ChatId& chat_id, vector<ChatMessage>& messages)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return false;
    }

    if (!m_db->get_chat_history(chat_id, messages))
    {
      return false;
    }

    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool ODAISdk::generate_streaming_chat_response(const ChatId& chat_id, const string& query,
                                               const GeneratorConfig& generator_config,
                                               OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {
    // Sanity check: SDK initialization
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
      return false;
    }

    if (query.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
      return false;
    }

    if (!generator_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid generator config passed");
      return false;
    }

    if (callback == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid callback passed");
      return false;
    }

    // Call the RAG engine's generate_streaming_chat_response method
    int32_t total_tokens =
        m_rag_engine->generate_streaming_chat_response(chat_id, query, generator_config, callback, user_data);

    if (total_tokens < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming chat response for chat_id: {}", chat_id);
      return false;
    }

    ODAI_LOG(ODAI_LOG_INFO,
             "Successfully generated streaming chat response for chat_id: {} "
             "with {} tokens",
             chat_id, total_tokens);

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught: {}", e.what());
    return false;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught");
    return false;
  }
}

bool ODAISdk::unload_chat(const ChatId& chat_id)
{
  try
  {
    if (!m_sdk_initialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return false;
    }

    return m_rag_engine->unload_chat_session(chat_id);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}
