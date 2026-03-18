#include "odai_sdk.h"
#include "audioEngine/odai_miniaudio_decoder.h"
#include "imageEngine/odai_stb_image_decoder.h"
#include "ragEngine/odai_rag_engine.h"

#include "types/odai_common_types.h"
#include "types/odai_types.h"
#include "utils/odai_helpers.h"

#include <cstdint>
#include <memory>

OdaiLogger* get_odai_logger()
{
  return OdaiSdk::get_instance().get_logger();
}

OdaiSdk& OdaiSdk::get_instance()
{
  static OdaiSdk instance;
  return instance;
}

std::unique_ptr<IOdaiAudioDecoder> OdaiSdk::get_new_odai_audio_decoder_instance()
{
#ifdef ODAI_ENABLE_MINIAUDIO
  return std::make_unique<OdaiMiniAudioDecoder>();
#else
  return nullptr;
#endif
}

std::unique_ptr<IOdaiImageDecoder> OdaiSdk::get_new_odai_image_decoder_instance()
{
#ifdef ODAI_ENABLE_STB_IMAGE
  return std::make_unique<OdaiStbImageDecoder>();
#else
  return nullptr;
#endif
}

OdaiSdk::OdaiSdk()
{
  m_logger = std::make_unique<OdaiLogger>();
}

OdaiSdk::~OdaiSdk()
{
  // Clean up if needed, though smart pointers in globals handle themselves
  // mostly But we might want to shut down explicitly if order matters
}

void OdaiSdk::set_logger(OdaiLogCallbackFn callback, void* user_data)
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

void OdaiSdk::set_log_level(OdaiLogLevel log_level)
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

bool OdaiSdk::initialize_sdk(const DBConfig& db_config, const BackendEngineConfig& backend_config)
{
  try
  {
    if (!db_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid db config passed");
      return false;
    }

    if (!backend_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid backend engine config passed");
      return false;
    }

    // Initalize the RAGEngine
    m_ragEngine = std::make_unique<OdaiRagEngine>(db_config, backend_config);

    if ((m_ragEngine == nullptr))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize RAG engine");
      m_sdkInitialized = false;
      return false;
    }

    m_sdkInitialized = true;
    ODAI_LOG(ODAI_LOG_INFO, "ODAI SDK Initialized successfully");
    return true;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    m_sdkInitialized = false;
    return false;
  }
}

OdaiResult<void> OdaiSdk::register_model_files(const ModelName& name, const ModelFiles& files)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
    }

    return m_ragEngine->register_model_files(name, files);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
}

OdaiResult<void> OdaiSdk::update_model_files(const ModelName& name, const ModelFiles& files, UpdateModelFlag flag)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
    }

    return m_ragEngine->update_model_files(name, files, flag);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
}

bool OdaiSdk::create_semantic_space(const SemanticSpaceConfig& config)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (!config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid semantic space config passed");
      return false;
    }

    // TODO: if dim == 0 then auto infer from model

    return m_ragEngine->create_semantic_space(config);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool OdaiSdk::get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_ragEngine->get_semantic_space_config(name, config);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool OdaiSdk::list_semantic_spaces(std::vector<SemanticSpaceConfig>& spaces)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_ragEngine->list_semantic_spaces(spaces);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool OdaiSdk::delete_semantic_space(const SemanticSpaceName& name)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    return m_ragEngine->delete_semantic_space(name);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return false;
  }
}

bool OdaiSdk::add_document(const std::string& content, const DocumentId& document_id,
                           const SemanticSpaceName& semantic_space_name, const ScopeId& scope_id) const
{
  try
  {
    if (!m_sdkInitialized)
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

int32_t OdaiSdk::generate_streaming_response(const LLMModelConfig& llm_model_config,
                                             const std::vector<InputItem>& prompt, const SamplerConfig& sampler_config,
                                             OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {

    if (!m_sdkInitialized)
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

    if (prompt.empty())
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
        m_ragEngine->generate_streaming_response(llm_model_config, prompt, sampler_config, callback, user_data);
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

bool OdaiSdk::create_chat(const ChatId& chat_id_in, const ChatConfig& chat_config, ChatId& chat_id_out)
{
  try
  {
    if (!m_sdkInitialized)
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
      if (m_ragEngine->chat_id_exists(chat_id))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} already exists", chat_id_in);
        return false;
      }
    }

    if (!m_ragEngine->create_chat(chat_id, chat_config))
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

bool OdaiSdk::get_chat_history(const ChatId& chat_id, std::vector<ChatMessage>& messages)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return false;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return false;
    }

    if (!m_ragEngine->get_chat_history(chat_id, messages))
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

int32_t OdaiSdk::generate_streaming_chat_response(const ChatId& chat_id, const std::vector<InputItem>& prompt,
                                                  const GeneratorConfig& generator_config,
                                                  OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {
    // Sanity check: SDK initialization
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return -1;
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
      return -1;
    }

    if (prompt.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
      return -1;
    }

    if (!generator_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid generator config passed");
      return -1;
    }

    if (callback == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid callback passed");
      return -1;
    }

    // Call the RAG engine's generate_streaming_chat_response method
    int32_t total_tokens =
        m_ragEngine->generate_streaming_chat_response(chat_id, prompt, generator_config, callback, user_data);

    if (total_tokens < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming chat response for chat_id: {}", chat_id);
      return -1;
    }

    ODAI_LOG(ODAI_LOG_INFO,
             "Successfully generated streaming chat response for chat_id: {} "
             "with {} tokens",
             chat_id, total_tokens);

    return total_tokens;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught: {}", e.what());
    return -1;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught");
    return -1;
  }
}