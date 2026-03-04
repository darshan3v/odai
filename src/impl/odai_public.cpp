#include "odai_public.h"
#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_ctypes.h"
#include "types/odai_type_conversions.h"
#include "utils/odai_csanitizers.h"

using namespace std;

void odai_set_logger(OdaiLogCallbackFn callback, void* user_data)
{
  OdaiSdk::get_instance().set_logger(callback, user_data);
}

void odai_set_log_level(OdaiLogLevel log_level)
{
  OdaiSdk::get_instance().set_log_level(log_level);
}

bool odai_initialize_sdk(const c_DbConfig* c_db_config, const c_BackendEngineConfig* c_backend_engine_config)
{
  if (!is_sane(c_db_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid dbConfig passed");
    return false;
  }

  if (!is_sane(c_backend_engine_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid backendEngineConfig passed");
    return false;
  }

  return OdaiSdk::get_instance().initialize_sdk(to_cpp(*c_db_config), to_cpp(*c_backend_engine_config));
}

bool odai_register_model_files(const c_ModelName model_name, const c_ModelFiles* files)
{
  if (model_name == nullptr || files == nullptr || !is_sane(files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_register_model_files");
    return false;
  }

  return OdaiSdk::get_instance().register_model_files(ModelName(model_name), to_cpp(*files));
}

bool odai_update_model_files(const c_ModelName model_name, const c_ModelFiles* files, c_UpdateModelFlag flag)
{
  if (model_name == nullptr || files == nullptr || !is_sane(files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_update_model_files");
    return false;
  }
  return OdaiSdk::get_instance().update_model_files(ModelName(model_name), to_cpp(*files),
                                                    to_cpp_update_model_flag(flag));
}

bool odai_create_semantic_space(const c_SemanticSpaceConfig* config)
{
  if (!is_sane(config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid semantic space config passed");
    return false;
  }
  return OdaiSdk::get_instance().create_semantic_space(to_cpp(*config));
}

bool odai_get_semantic_space(const c_SemanticSpaceName semantic_space_name, c_SemanticSpaceConfig* config_out)
{
  if (semantic_space_name == nullptr || config_out == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid arguments passed to odai_get_semantic_space");
    return false;
  }

  SemanticSpaceConfig config;
  if (!OdaiSdk::get_instance().get_semantic_space_config(SemanticSpaceName(semantic_space_name), config))
  {
    return false;
  }

  *config_out = to_c(config);
  return true;
}

void odai_free_semantic_space_config(c_SemanticSpaceConfig* config)
{
  if (config == nullptr)
  {
    return;
  }
  free_members(config);
}

bool odai_list_semantic_spaces(c_SemanticSpaceConfig** spaces_out, size_t* spaces_count)
{
  if (spaces_out == nullptr || spaces_count == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
    return false;
  }

  vector<SemanticSpaceConfig> spaces;
  if (!OdaiSdk::get_instance().list_semantic_spaces(spaces))
  {
    *spaces_out = nullptr;
    *spaces_count = 0;
    return false;
  }

  if (spaces.empty())
  {
    *spaces_out = nullptr;
    *spaces_count = 0;
    return true;
  }

  *spaces_count = spaces.size();
  *spaces_out = static_cast<c_SemanticSpaceConfig*>(malloc(sizeof(c_SemanticSpaceConfig) * (*spaces_count)));

  if (*spaces_out == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for spaces list");
    *spaces_count = 0;
    return false;
  }

  for (size_t i = 0; i < spaces.size(); ++i)
  {
    (*spaces_out)[i] = to_c(spaces[i]);
  }

  return true;
}

void odai_free_semantic_spaces_list(c_SemanticSpaceConfig* spaces, size_t count)
{
  if (spaces == nullptr)
  {
    return;
  }

  try
  {
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

bool odai_delete_semantic_space(const c_SemanticSpaceName name)
{
  if (name == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid space name passed");
    return false;
  }
  return OdaiSdk::get_instance().delete_semantic_space(string(name));
}

bool odai_add_document(const char* content, const c_DocumentId document_id,
                       const c_SemanticSpaceName semantic_space_name, const c_ScopeId scope_id)
{
  if (content == nullptr || document_id == nullptr || semantic_space_name == nullptr || scope_id == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_add_document");
    return false;
  }

  return OdaiSdk::get_instance().add_document(string(content), DocumentId(document_id),
                                              SemanticSpaceName(semantic_space_name), ScopeId(scope_id));
}

int32_t odai_generate_streaming_response(const c_LlmModelConfig* llm_model_config, const c_InputItem* c_prompt_items,
                                         size_t prompt_items_count, const c_SamplerConfig* c_sampler_config,
                                         OdaiStreamRespCallbackFn c_callback, void* c_user_data)
{

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

  if (c_prompt_items == nullptr || prompt_items_count == 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
    return -1;
  }

  vector<InputItem> prompt_items;
  for (size_t i = 0; i < prompt_items_count; ++i)
  {
    if (!is_sane(&c_prompt_items[i]))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid input item at index {}", i);
      return -1;
    }
    prompt_items.push_back(to_cpp(c_prompt_items[i]));
  }

  return OdaiSdk::get_instance().generate_streaming_response(to_cpp(*llm_model_config), prompt_items,
                                                             to_cpp(*c_sampler_config), c_callback, c_user_data);
}

bool odai_create_chat(const c_ChatId c_chat_id_in, const c_ChatConfig* c_chat_config, c_ChatId* c_chat_id_out)
{
  if (!is_sane(c_chat_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
    return false;
  }

  if (c_chat_id_out == nullptr && c_chat_id_in == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid parameters passed: c_chat_id_out and c_chat_id_in are null");
    return false;
  }

  ChatId chat_id_in = (c_chat_id_in != nullptr) ? ChatId(c_chat_id_in) : ChatId("");
  ChatId chat_id_out;

  bool result = OdaiSdk::get_instance().create_chat(chat_id_in, to_cpp(*c_chat_config), chat_id_out);

  if (result)
  {
    *c_chat_id_out = strdup(chat_id_out.c_str());
    if (*c_chat_id_out == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for chat_id_out");
      return false;
    }
  }
  else
  {
    *c_chat_id_out = nullptr;
  }

  return result;
}

void odai_free_chat_id(c_ChatId chat_id)
{
  if (chat_id != nullptr)
  {
    free(chat_id);
  }
}

bool odai_load_chat(const c_ChatId c_chat_id)
{
  if (c_chat_id == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
    return false;
  }
  return OdaiSdk::get_instance().load_chat(ChatId(c_chat_id));
}

bool odai_get_chat_history(const c_ChatId c_chat_id, c_ChatMessage** c_messages_out, size_t* messages_count)
{
  if (c_chat_id == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
    return false;
  }

  if (c_messages_out == nullptr || messages_count == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
    return false;
  }

  vector<ChatMessage> messages;
  if (!OdaiSdk::get_instance().get_chat_history(ChatId(c_chat_id), messages))
  {
    *c_messages_out = nullptr;
    *messages_count = 0;
    return false; // Or true if empty? Original returned false on DB error, true on empty.
                  // OdaiSdk::get_chat_history returns false on error.
  }

  if (messages.empty())
  {
    *c_messages_out = nullptr;
    *messages_count = 0;
    return true;
  }

  *messages_count = messages.size();
  *c_messages_out = static_cast<c_ChatMessage*>(malloc(sizeof(c_ChatMessage) * (*messages_count)));

  if (*c_messages_out == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for messages");
    *messages_count = 0;
    return false;
  }

  for (size_t i = 0; i < messages.size(); ++i)
  {
    (*c_messages_out)[i] = to_c(messages[i]);
  }

  return true;
}

void odai_free_chat_messages(c_ChatMessage* c_messages, size_t count)
{
  if (c_messages == nullptr)
  {
    return;
  }

  try
  {
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

bool odai_generate_streaming_chat_response(const c_ChatId c_chat_id, const c_InputItem* c_prompt_items,
                                           size_t prompt_items_count, const c_GeneratorConfig* c_generator_config,
                                           OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (c_chat_id == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
    return false;
  }

  if (c_prompt_items == nullptr || prompt_items_count == 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
    return false;
  }

  if (!is_sane(c_generator_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid generator config passed");
    return false;
  }

  vector<InputItem> prompt_items;
  for (size_t i = 0; i < prompt_items_count; ++i)
  {
    if (!is_sane(&c_prompt_items[i]))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid input item at index {}", i);
      return false;
    }
    prompt_items.push_back(to_cpp(c_prompt_items[i]));
  }

  return OdaiSdk::get_instance().generate_streaming_chat_response(ChatId(c_chat_id), prompt_items,
                                                                  to_cpp(*c_generator_config), callback, user_data);
}

bool odai_unload_chat(const c_ChatId c_chat_id)
{
  if (c_chat_id == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
    return false;
  }
  return OdaiSdk::get_instance().unload_chat(ChatId(c_chat_id));
}