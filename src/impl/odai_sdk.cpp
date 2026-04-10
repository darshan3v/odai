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

OdaiResult<void> OdaiSdk::initialize_sdk(const DBConfig& db_config, const BackendEngineConfig& backend_config)
{
  try
  {
    if (!db_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid db config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (!backend_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid backend engine config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // Initalize the RAGEngine
    m_ragEngine = std::make_unique<OdaiRagEngine>(db_config, backend_config);

    if ((m_ragEngine == nullptr))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to create RAG engine");
      m_sdkInitialized = false;
      return unexpected_internal_error<void>();
    }

    OdaiResult<void> init_res = m_ragEngine->initialize_rag_engine();
    if (!init_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize RAG engine, error code: {}",
               static_cast<std::uint32_t>(init_res.error()));
      m_sdkInitialized = false;
      return tl::unexpected(init_res.error());
    }

    m_sdkInitialized = true;
    ODAI_LOG(ODAI_LOG_INFO, "ODAI SDK Initialized successfully");
    return {};
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    m_sdkInitialized = false;
    return unexpected_internal_error<void>();
  }
}

OdaiResult<void> OdaiSdk::register_model_files(const ModelName& name, const ModelFiles& files)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<void>();
    }

    return m_ragEngine->register_model_files(name, files);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<void>();
  }
}

OdaiResult<void> OdaiSdk::update_model_files(const ModelName& name, const ModelFiles& files, UpdateModelFlag flag)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<void>();
    }

    return m_ragEngine->update_model_files(name, files, flag);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<void>();
  }
}

OdaiResult<void> OdaiSdk::create_semantic_space(const SemanticSpaceConfig& config)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<void>();
    }

    if (!config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid semantic space config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // TODO: if dim == 0 then auto infer from model

    OdaiResult<void> res = m_ragEngine->create_semantic_space(config);
    if (!res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to create semantic space, error code: {}",
               static_cast<std::uint32_t>(res.error()));
      return tl::unexpected(res.error());
    }

    return {};
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<void>();
  }
}

OdaiResult<SemanticSpaceConfig> OdaiSdk::get_semantic_space_config(const SemanticSpaceName& name)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<SemanticSpaceConfig>();
    }

    OdaiResult<SemanticSpaceConfig> res = m_ragEngine->get_semantic_space_config(name);
    if (!res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to get semantic space config, error code: {}",
               static_cast<std::uint32_t>(res.error()));
      return tl::unexpected(res.error());
    }

    return res;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<SemanticSpaceConfig>();
  }
}

OdaiResult<std::vector<SemanticSpaceConfig>> OdaiSdk::list_semantic_spaces()
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<std::vector<SemanticSpaceConfig>>();
    }

    OdaiResult<std::vector<SemanticSpaceConfig>> res = m_ragEngine->list_semantic_spaces();
    if (!res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to list semantic spaces, error code: {}",
               static_cast<std::uint32_t>(res.error()));
      return tl::unexpected(res.error());
    }

    return res;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<std::vector<SemanticSpaceConfig>>();
  }
}

OdaiResult<void> OdaiSdk::delete_semantic_space(const SemanticSpaceName& name)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<void>();
    }

    OdaiResult<void> res = m_ragEngine->delete_semantic_space(name);
    if (!res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to delete semantic space, error code: {}",
               static_cast<std::uint32_t>(res.error()));
      return tl::unexpected(res.error());
    }

    return {};
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<void>();
  }
}

OdaiResult<void> OdaiSdk::add_document(const std::string& content, const DocumentId& document_id,
                                       const SemanticSpaceName& semantic_space_name, const ScopeId& scope_id) const
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<void>();
    }

    if (content.empty() || document_id.empty() || semantic_space_name.empty() || scope_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid document arguments passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // ToDo : call rag engine to add document

    ODAI_LOG(ODAI_LOG_INFO, "Adding document to space: {}", semantic_space_name);

    return {};
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<void>();
  }
}

OdaiResult<StreamingStats> OdaiSdk::generate_streaming_response(const LLMModelConfig& llm_model_config,
                                                                const std::vector<InputItem>& prompt,
                                                                const SamplerConfig& sampler_config,
                                                                OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {

    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<StreamingStats>();
    }

    if (!llm_model_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid LLM Model Config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (!sampler_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid Sampler Config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (prompt.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (callback == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "empty callback passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    OdaiResult<StreamingStats> stream_res =
        m_ragEngine->generate_streaming_response(llm_model_config, prompt, sampler_config, callback, user_data);
    if (!stream_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to generate response, error code: {}",
               static_cast<std::uint32_t>(stream_res.error()));
      return tl::unexpected(stream_res.error());
    }

    return stream_res;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<StreamingStats>();
  }
}

OdaiResult<ChatId> OdaiSdk::create_chat(const ChatId& chat_id_in, const ChatConfig& chat_config)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<ChatId>();
    }

    ChatId chat_id;

    if (!chat_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // if chat_id is empty then we will generate and set the out, else we will
    // use the given chat_id, given it's unique
    if (chat_id_in.empty())
    {
      chat_id = generate_chat_id();
    }
    else
    {
      chat_id = chat_id_in;
      OdaiResult<bool> exists_res = m_ragEngine->chat_id_exists(chat_id);
      if (!exists_res)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to check chat existence, error code: {}",
                 static_cast<std::uint32_t>(exists_res.error()));
        return tl::unexpected(exists_res.error());
      }

      if (exists_res.value())
      {
        ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} already exists", chat_id_in);
        return tl::unexpected(OdaiResultEnum::ALREADY_EXISTS);
      }
    }

    OdaiResult<void> create_res = m_ragEngine->create_chat(chat_id, chat_config);
    if (!create_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to create chat, error code: {}", static_cast<std::uint32_t>(create_res.error()));
      return tl::unexpected(create_res.error());
    }

    return chat_id;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<ChatId>();
  }
}

OdaiResult<std::vector<ChatMessage>> OdaiSdk::get_chat_history(const ChatId& chat_id)
{
  try
  {
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<std::vector<ChatMessage>>();
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    OdaiResult<std::vector<ChatMessage>> res = m_ragEngine->get_chat_history(chat_id);
    if (!res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat history, error code: {}", static_cast<std::uint32_t>(res.error()));
      return tl::unexpected(res.error());
    }

    return res;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    return unexpected_internal_error<std::vector<ChatMessage>>();
  }
}

OdaiResult<StreamingStats> OdaiSdk::generate_streaming_chat_response(const ChatId& chat_id,
                                                                     const std::vector<InputItem>& prompt,
                                                                     const GeneratorConfig& generator_config,
                                                                     OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {
    // Sanity check: SDK initialization
    if (!m_sdkInitialized)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
      return unexpected_not_initialized<StreamingStats>();
    }

    if (chat_id.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (prompt.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (!generator_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid generator config passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    if (callback == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid callback passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // Call the RAG engine's generate_streaming_chat_response method
    OdaiResult<StreamingStats> stream_res =
        m_ragEngine->generate_streaming_chat_response(chat_id, prompt, generator_config, callback, user_data);

    if (!stream_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming chat response for chat_id: {}, error code: {}", chat_id,
               static_cast<std::uint32_t>(stream_res.error()));
      return tl::unexpected(stream_res.error());
    }

    ODAI_LOG(ODAI_LOG_INFO,
             "Successfully generated streaming chat response for chat_id: {} "
             "with {} tokens",
             chat_id, stream_res->m_generatedTokens);

    return stream_res;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught: {}", e.what());
    return unexpected_internal_error<StreamingStats>();
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught");
    return unexpected_internal_error<StreamingStats>();
  }
}
