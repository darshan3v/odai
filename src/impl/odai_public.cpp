#include "odai_public.h"
#include "odai_sdk.h"
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

bool odai_initialize_rag_engine(const c_RagConfig *c_config)
{
    if (!is_sane(c_config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid ragconfig passed");
        return false;
    }
    
    // Check inner configs sanity as done in original code, though ODAISdk also checks logic
    if (!is_sane(&c_config->embeddingModelConfig) || !is_sane(&c_config->llmModelConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "make sure given ragconfig is correct");
        return false;
    }

    return ODAISdk::get_instance().initialize_rag_engine(toCpp(*c_config));
}

bool odai_add_document(const char *content, const c_DocumentId document_id, const c_ScopeId scope_id)
{
    if (content == nullptr || document_id == nullptr || scope_id == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid arguments passed to odai_add_document");
        return false;
    }
    return ODAISdk::get_instance().add_document(string(content), DocumentId(document_id), ScopeId(scope_id));
}

int32_t odai_generate_streaming_response(const char *c_query,
                                         odai_stream_resp_callback_fn c_callback, void *c_user_data)
{
    if (c_query == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid query passed");
        return -1;
    }

    return ODAISdk::get_instance().generate_streaming_response(string(c_query), c_callback, c_user_data);
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
            if (c_messages[i].content != nullptr)
            {
                free(c_messages[i].content);
                c_messages[i].content = nullptr;
            }
            if (c_messages[i].message_metadata != nullptr)
            {
                free(c_messages[i].message_metadata);
                c_messages[i].message_metadata = nullptr;
            }
        }
        free(c_messages);
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while freeing chat messages");
    }
}

bool odai_generate_streaming_chat_response(const c_ChatId c_chat_id, const char *c_query, const c_ScopeId c_scope_id,
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

    return ODAISdk::get_instance().generate_streaming_chat_response(ChatId(c_chat_id), string(c_query), (c_scope_id ? ScopeId(c_scope_id) : ScopeId("")), callback, user_data);
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