#include "odai_public.h"
#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_ctypes.h"
#include "types/odai_result.h"
#include "types/odai_type_conversions.h"
#include "utils/odai_csanitizers.h"
#include "utils/odai_exception_macros.h"
#include <cstdint>

void odai_set_logger(OdaiLogCallbackFn callback, void* user_data)
{
  try
  {
    OdaiSdk::get_instance().set_logger(callback, user_data);
  }
  ODAI_CATCH_LOG()
}

void odai_set_log_level(OdaiLogLevel log_level)
{
  try
  {
    OdaiSdk::get_instance().set_log_level(log_level);
  }
  ODAI_CATCH_LOG()
}

c_OdaiResult odai_initialize_sdk(const c_DbConfig* c_db_config, const c_BackendEngineConfig* c_backend_engine_config)
{
  try
  {
    if (!is_sane(c_db_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid dbConfig passed");
      return ODAI_INVALID_ARGUMENT;
    }

    if (!is_sane(c_backend_engine_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid backendEngineConfig passed");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res =
        OdaiSdk::get_instance().initialize_sdk(to_cpp(*c_db_config), to_cpp(*c_backend_engine_config));
    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

c_OdaiResult odai_register_model_files(const c_ModelName model_name, const c_ModelFiles* files)
{
  try
  {
    if (model_name == nullptr || files == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_register_model_files");
      return ODAI_INVALID_ARGUMENT;
    }

    if (!is_sane(files))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_register_model_files");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res = OdaiSdk::get_instance().register_model_files(ModelName(model_name), to_cpp(*files));
    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

c_OdaiResult odai_update_model_files(const c_ModelName model_name, const c_ModelFiles* files, c_UpdateModelFlag flag)
{
  try
  {
    if (model_name == nullptr || files == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_update_model_files");
      return ODAI_INVALID_ARGUMENT;
    }

    if (!is_sane(files))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_update_model_files");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res = OdaiSdk::get_instance().update_model_files(ModelName(model_name), to_cpp(*files),
                                                                      to_cpp_update_model_flag(flag));

    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

c_OdaiResult odai_create_semantic_space(const c_SemanticSpaceConfig* config)
{
  try
  {
    if (!is_sane(config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid semantic space config passed");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res = OdaiSdk::get_instance().create_semantic_space(to_cpp(*config));
    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

c_OdaiResult odai_get_semantic_space(const c_SemanticSpaceName semantic_space_name, c_SemanticSpaceConfig* config_out)
{
  try
  {
    if (semantic_space_name == nullptr || config_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid arguments passed to odai_get_semantic_space");
      return ODAI_INVALID_ARGUMENT;
    }

    *config_out = {};

    OdaiResult<SemanticSpaceConfig> res =
        OdaiSdk::get_instance().get_semantic_space_config(SemanticSpaceName(semantic_space_name));
    if (!res)
    {
      return to_c_result(res.error());
    }

    *config_out = to_c(res.value());
    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

void odai_free_semantic_space_config(c_SemanticSpaceConfig* config)
{
  try
  {
    if (config == nullptr)
    {
      return;
    }
    free_members(config);
  }
  ODAI_CATCH_LOG()
}

c_OdaiResult odai_list_semantic_spaces(c_SemanticSpaceConfig** spaces_out, uint16_t* spaces_count)
{
  try
  {
    if (spaces_out == nullptr || spaces_count == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
      return ODAI_INVALID_ARGUMENT;
    }

    *spaces_out = nullptr;
    *spaces_count = 0;

    OdaiResult<std::vector<SemanticSpaceConfig>> res = OdaiSdk::get_instance().list_semantic_spaces();
    if (!res)
    {
      return to_c_result(res.error());
    }

    const std::vector<SemanticSpaceConfig>& spaces = res.value();

    if (spaces.empty())
    {
      return ODAI_SUCCESS;
    }

    *spaces_count = spaces.size();
    *spaces_out = static_cast<c_SemanticSpaceConfig*>(malloc(sizeof(c_SemanticSpaceConfig) * (*spaces_count)));

    if (*spaces_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for spaces list");
      *spaces_count = 0;
      return ODAI_INTERNAL_ERROR;
    }

    size_t converted_count = 0;
    try
    {
      for (; converted_count < spaces.size(); ++converted_count)
      {
        (*spaces_out)[converted_count] = to_c(spaces[converted_count]);
      }
    }
    catch (...)
    {
      for (size_t i = 0; i < converted_count; ++i)
      {
        free_members(&(*spaces_out)[i]);
      }
      free(*spaces_out);
      *spaces_out = nullptr;
      *spaces_count = 0;
      throw;
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

void odai_free_semantic_spaces_list(c_SemanticSpaceConfig* spaces, uint16_t count)
{
  try
  {
    if (spaces == nullptr)
    {
      return;
    }
    for (size_t i = 0; i < count; ++i)
    {
      free_members(&spaces[i]);
    }
    free(spaces);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while freeing semantic spaces");
  }
}

c_OdaiResult odai_delete_semantic_space(const c_SemanticSpaceName name)
{
  try
  {
    if (name == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid space name passed");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res = OdaiSdk::get_instance().delete_semantic_space(std::string(name));
    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

c_OdaiResult odai_add_document(const char* content, const c_DocumentId document_id,
                               const c_SemanticSpaceName semantic_space_name, const c_ScopeId scope_id)
{
  try
  {
    if (content == nullptr || document_id == nullptr || semantic_space_name == nullptr || scope_id == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_add_document");
      return ODAI_INVALID_ARGUMENT;
    }

    OdaiResult<void> res = OdaiSdk::get_instance().add_document(
        std::string(content), DocumentId(document_id), SemanticSpaceName(semantic_space_name), ScopeId(scope_id));
    if (!res)
    {
      return to_c_result(res.error());
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

int32_t odai_generate_streaming_response(const c_LlmModelConfig* llm_model_config, const c_InputItem* c_prompt_items,
                                         uint16_t prompt_items_count, const c_SamplerConfig* c_sampler_config,
                                         OdaiStreamRespCallbackFn c_callback, void* c_user_data)
{
  try
  {
    if (c_prompt_items == nullptr || prompt_items_count == 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
      return -1;
    }

    if (!is_sane(llm_model_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid llm_model_config passed");
      return -1;
    }

    if (!is_sane(c_sampler_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid sampler_config passed");
      return -1;
    }

    std::vector<InputItem> prompt_items;
    for (size_t i = 0; i < prompt_items_count; ++i)
    {
      if (!is_sane(&c_prompt_items[i]))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid input item at index {}", i);
        return -1;
      }

      prompt_items.push_back(to_cpp(c_prompt_items[i]));
    }

    OdaiResult<StreamingStats> res = OdaiSdk::get_instance().generate_streaming_response(
        to_cpp(*llm_model_config), prompt_items, to_cpp(*c_sampler_config), c_callback, c_user_data);
    if (!res)
    {
      return -1;
    }

    return res->m_generatedTokens;
  }
  ODAI_CATCH_RETURN(-1)
}

c_OdaiResult odai_create_chat(const c_ChatId c_chat_id_in, const c_ChatConfig* c_chat_config, c_ChatId* c_chat_id_out)
{
  try
  {
    if (c_chat_id_out != nullptr)
    {
      *c_chat_id_out = nullptr;
    }

    if (c_chat_id_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameter passed to odai_create_chat");
      return ODAI_INVALID_ARGUMENT;
    }

    if (!is_sane(c_chat_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
      return ODAI_INVALID_ARGUMENT;
    }

    ChatId chat_id_in = (c_chat_id_in != nullptr) ? ChatId(c_chat_id_in) : ChatId("");
    OdaiResult<ChatId> res = OdaiSdk::get_instance().create_chat(chat_id_in, to_cpp(*c_chat_config));

    if (!res)
    {
      return to_c_result(res.error());
    }

    *c_chat_id_out = strdup(res.value().c_str());
    if (*c_chat_id_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for chat_id_out");
      return ODAI_INTERNAL_ERROR;
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

void odai_free_chat_id(c_ChatId chat_id)
{
  try
  {
    if (chat_id != nullptr)
    {
      free(chat_id);
    }
  }
  ODAI_CATCH_LOG()
}

c_OdaiResult odai_get_chat_history(const c_ChatId c_chat_id, c_ChatMessage** c_messages_out, uint16_t* messages_count)
{
  try
  {
    if (c_chat_id == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
      return ODAI_INVALID_ARGUMENT;
    }

    if (c_messages_out == nullptr || messages_count == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
      return ODAI_INVALID_ARGUMENT;
    }

    *c_messages_out = nullptr;
    *messages_count = 0;

    OdaiResult<std::vector<ChatMessage>> res = OdaiSdk::get_instance().get_chat_history(ChatId(c_chat_id));
    if (!res)
    {
      return to_c_result(res.error());
    }

    const std::vector<ChatMessage>& messages = res.value();

    if (messages.empty())
    {
      return ODAI_SUCCESS;
    }

    *messages_count = messages.size();
    *c_messages_out = static_cast<c_ChatMessage*>(malloc(sizeof(c_ChatMessage) * (*messages_count)));

    if (*c_messages_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for messages");
      *messages_count = 0;
      return ODAI_INTERNAL_ERROR;
    }

    size_t converted_count = 0;
    try
    {
      for (; converted_count < messages.size(); ++converted_count)
      {
        (*c_messages_out)[converted_count] = to_c(messages[converted_count]);
      }
    }
    catch (...)
    {
      for (size_t i = 0; i < converted_count; ++i)
      {
        free_members(&(*c_messages_out)[i]);
      }
      free(*c_messages_out);
      *c_messages_out = nullptr;
      *messages_count = 0;
      throw;
    }

    return ODAI_SUCCESS;
  }
  ODAI_CATCH_RETURN(ODAI_INTERNAL_ERROR)
}

void odai_free_chat_messages(c_ChatMessage* c_messages, uint16_t count)
{
  try
  {
    if (c_messages == nullptr)
    {
      return;
    }

    for (size_t i = 0; i < count; ++i)
    {
      free_members(&c_messages[i]);
    }
    free(c_messages);
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while freeing chat messages");
  }
}

int32_t odai_generate_streaming_chat_response(const c_ChatId c_chat_id, const c_InputItem* c_prompt_items,
                                              uint16_t prompt_items_count, const c_GeneratorConfig* c_generator_config,
                                              OdaiStreamRespCallbackFn callback, void* user_data)
{
  try
  {
    if (c_chat_id == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
      return -1;
    }

    if (c_prompt_items == nullptr || prompt_items_count == 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
      return -1;
    }

    if (!is_sane(c_generator_config))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid generator config passed");
      return -1;
    }

    std::vector<InputItem> prompt_items;
    for (size_t i = 0; i < prompt_items_count; ++i)
    {
      if (!is_sane(&c_prompt_items[i]))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid input item at index {}", i);
        return -1;
      }
      prompt_items.push_back(to_cpp(c_prompt_items[i]));
    }

    OdaiResult<StreamingStats> res = OdaiSdk::get_instance().generate_streaming_chat_response(
        ChatId(c_chat_id), prompt_items, to_cpp(*c_generator_config), callback, user_data);
    if (!res)
    {
      return -1;
    }

    return res->m_generatedTokens;
  }
  ODAI_CATCH_RETURN(-1)
}
