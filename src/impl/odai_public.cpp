#include "odai_public.h"
#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_ctypes.h"
#include "types/odai_type_conversions.h"
#include "utils/odai_csanitizers.h"
#include "utils/string_utils.h"
#include "odai_sdk.h"

using namespace std;

void odai_set_logger(odai_log_callback_fn callback, void *user_data)
{
    ODAISdk::get_instance().set_logger(callback, user_data);
}

void odai_set_log_level(OdaiLogLevel log_level)
{
    ODAISdk::get_instance().set_log_level(log_level);
}

bool odai_initialize_sdk(const c_DBConfig *c_dbConfig, const c_BackendEngineConfig *c_backendEngineConfig)
{
    if (!is_sane(c_dbConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid dbConfig passed");
        return false;
    }

    if (!is_sane(c_backendEngineConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid backendEngineConfig passed");
        return false;
    }

    return ODAISdk::get_instance().initialize_sdk(toCpp(*c_dbConfig), toCpp(*c_backendEngineConfig));
}

bool odai_register_model(c_ModelName name, c_ModelPath path, c_ModelType type)
{
    if (name == nullptr || path == nullptr)
    {
         ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_register_model");
         return false;
    }

    if (!is_sane(type))
    {
         ODAI_LOG(ODAI_LOG_ERROR, "Invalid model type passed");
         return false;
    }

    return ODAISdk::get_instance().register_model(ModelName(name), ModelPath(path), toCpp(type));
}

bool odai_update_model_path(c_ModelName name, c_ModelPath path)
{
    if (name == nullptr || path == nullptr)
    {
         ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_update_model_path");
         return false;
    }
    return ODAISdk::get_instance().update_model_path(ModelName(name), ModelPath(path));
}

bool odai_create_semantic_space(const c_SemanticSpaceConfig *config)
{
    if (!is_sane(config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid semantic space config passed");
        return false;
    }
    return ODAISdk::get_instance().create_semantic_space(toCpp(*config));
}

bool odai_get_semantic_space(const c_SemanticSpaceName name, c_SemanticSpaceConfig *config_out)
{
    if (name == nullptr || config_out == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid arguments passed to odai_get_semantic_space");
        return false;
    }

    SemanticSpaceConfig config;
    if (!ODAISdk::get_instance().get_semantic_space_config(SemanticSpaceName(name), config))
    {
         return false;
    }

    *config_out = toC(config);
    return true;
}

void odai_free_semantic_space_config(c_SemanticSpaceConfig *config)
{
    if (config == nullptr) return;
    free_members(config);
}

bool odai_list_semantic_spaces(c_SemanticSpaceConfig **spaces_out, size_t *spaces_count)
{
    if (spaces_out == nullptr || spaces_count == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
        return false;
    }

    vector<SemanticSpaceConfig> spaces;
    if (!ODAISdk::get_instance().list_semantic_spaces(spaces))
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
    *spaces_out = static_cast<c_SemanticSpaceConfig *>(malloc(sizeof(c_SemanticSpaceConfig) * (*spaces_count)));

    if (*spaces_out == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for spaces list");
        *spaces_count = 0;
        return false;
    }

    for (size_t i = 0; i < spaces.size(); ++i)
    {
        (*spaces_out)[i] = toC(spaces[i]);
    }

    return true;
}

void odai_free_semantic_spaces_list(c_SemanticSpaceConfig *spaces, size_t count)
{
    if (spaces == nullptr)
        return;

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
    return ODAISdk::get_instance().delete_semantic_space(string(name));
}

bool odai_add_document(const char *content, const c_DocumentId document_id, const c_SemanticSpaceName semantic_space_name, const c_ScopeId scope_id)
{
    if (content == nullptr || document_id == nullptr || semantic_space_name == nullptr || scope_id == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_add_document");
        return false;
    }
    
    return ODAISdk::get_instance().add_document(string(content), DocumentId(document_id), SemanticSpaceName(semantic_space_name), ScopeId(scope_id));
}

int32_t odai_generate_streaming_response(const c_LLMModelConfig* llm_model_config, const char *c_query,
                                         odai_stream_resp_callback_fn c_callback, void *c_user_data)
{

    if(!is_sane(llm_model_config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid llm_model_config passed");
        return -1;
    }

    if (c_query == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
        return -1;
    }

    return ODAISdk::get_instance().generate_streaming_response(toCpp(*llm_model_config), string(c_query), c_callback, c_user_data);
}

bool odai_create_chat(const c_ChatId c_chat_id_in, const c_ChatConfig *c_chat_config, c_ChatId c_chat_id_out, size_t *chat_id_out_len)
{
    if (!is_sane(c_chat_config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
        return false;
    }

    ChatId chatIdIn = (c_chat_id_in != nullptr) ? ChatId(c_chat_id_in) : ChatId("");
    ChatId chatIdOut;

    bool result = ODAISdk::get_instance().create_chat(chatIdIn, toCpp(*c_chat_config), chatIdOut);

    if (result)
    {
        set_cstr_and_len(chatIdOut, c_chat_id_out, chat_id_out_len);
    }

    return result;
}

bool odai_load_chat(const c_ChatId c_chat_id)
{
    if (c_chat_id == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
        return false;
    }
    return ODAISdk::get_instance().load_chat(ChatId(c_chat_id));
}

bool odai_get_chat_history(const c_ChatId c_chat_id, c_ChatMessage **c_messages_out, size_t *messages_count)
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
    if (!ODAISdk::get_instance().get_chat_history(ChatId(c_chat_id), messages))
    {
        *c_messages_out = nullptr;
        *messages_count = 0;
        return false; // Or true if empty? Original returned false on DB error, true on empty.
                      // ODAISdk::get_chat_history returns false on error.
    }

    if (messages.empty())
    {
        *c_messages_out = nullptr;
        *messages_count = 0;
        return true;
    }

    *messages_count = messages.size();
    *c_messages_out = static_cast<c_ChatMessage *>(malloc(sizeof(c_ChatMessage) * (*messages_count)));

    if (*c_messages_out == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for messages");
        *messages_count = 0;
        return false;
    }

    for (size_t i = 0; i < messages.size(); ++i)
    {
        (*c_messages_out)[i] = toC(messages[i]);
    }

    return true;
}

void odai_free_chat_messages(c_ChatMessage *c_messages, size_t count)
{
    if (c_messages == nullptr)
        return;

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

bool odai_generate_streaming_chat_response(const c_ChatId c_chat_id, const char *c_query, const c_SemanticSpaceName semantic_space_name, const c_ScopeId c_scope_id,
                                           odai_stream_resp_callback_fn callback, void *user_data)
{
    if (c_chat_id == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
        return false;
    }

    if (c_query == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
        return false;
    }

    return ODAISdk::get_instance().generate_streaming_chat_response(ChatId(c_chat_id), string(c_query), (semantic_space_name ? SemanticSpaceName(semantic_space_name) : SemanticSpaceName("")), (c_scope_id ? ScopeId(c_scope_id) : ScopeId("")), callback, user_data);
}

bool odai_unload_chat(const c_ChatId c_chat_id)
{
    if (c_chat_id == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
        return false;
    }
    return ODAISdk::get_instance().unload_chat(ChatId(c_chat_id));
}