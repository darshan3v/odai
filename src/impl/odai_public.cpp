#include "odai_public.h"
#include "ragEngine/odai_rag_engine.h"
#include "odai_logger.h"

#include "db/odai_sqlite_db.h"
#include "backendEngine/odai_llama_backend_engine.h"
#include "utils/string_utils.h"
#include "utils/odai_csanitizers.h"
#include "utils/odai_helpers.h"
#include "types/odai_type_conversions.h"

using namespace std;

// SDK initialization state
static bool sdk_initialized = false;

void odai_set_logger(odai_log_callback_fn callback, void *user_data)
{
    try
    {
        g_odaiLogger->odai_set_logger(callback, user_data);
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    }
}

void odai_set_log_level(OdaiLogLevel log_level)
{
    try
    {
        g_odaiLogger->odai_set_log_level(log_level);
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    }
}

bool odai_initialize_sdk(const c_DBConfig *c_dbConfig, const c_BackendEngineConfig *c_backendEngineConfig)
{
    try
    {
        if (!is_sane(c_dbConfig))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid dbConfig passed");
            sdk_initialized = false;
            return false;
        }

        if (!is_sane(c_backendEngineConfig))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid backendEngineConfig passed");
            sdk_initialized = false;
            return false;
        }

        // Initialize the database backend based on dbType
        DBConfig db_config = toCpp(*c_dbConfig);

        if (db_config.dbType == SQLITE_DB)
        {
            g_db.reset(new ODAISqliteDb(db_config));
        }
        else
        {
            ODAI_LOG(ODAI_LOG_ERROR, "unsupported database type: {}", static_cast<int>(db_config.dbType));
            sdk_initialized = false;
            return false;
        }

        if ((g_db.get() == nullptr) || (!g_db->initialize_db()))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize db");
            sdk_initialized = false;
            return false;
        }

        // Initialize the backend engine based on engineType
        BackendEngineConfig backend_engine_config = toCpp(*c_backendEngineConfig);

        if (backend_engine_config.engineType == LLAMA_BACKEND_ENGINE)
            g_backendEngine.reset(new ODAILlamaEngine());
        else
        {
            ODAI_LOG(ODAI_LOG_ERROR, "unsupported backend engine type: {}", static_cast<int>(backend_engine_config.engineType));
            sdk_initialized = false;
            return false;
        }

        if ((g_backendEngine.get() == nullptr) || (!g_backendEngine->initialize_engine()))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize backend engine");
            sdk_initialized = false;
            return false;
        }

        sdk_initialized = true;
        ODAI_LOG(ODAI_LOG_INFO, "ODAI SDK Initialized successfully");
        return true;
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
        sdk_initialized = false;
        return false;
    }
}

bool odai_initialize_rag_engine(const c_RagConfig *c_config)
{
    try
    {
        if (!is_sane(c_config))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid ragconfig passed");
            return false;
        }

        g_ragEngine.reset(new ODAIRagEngine());

        if (!is_sane(&c_config->embeddingModelConfig) || !is_sane(&c_config->llmModelConfig))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "make sure given ragconfig is correct");
            return false;
        }

        if (!g_ragEngine->initialize(toCpp(*c_config)))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "failed to initalize RAG Engine");
            return false;
        }
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
        return false;
    }

    ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully initialized");
    return true;
}

bool odai_add_document(ODAIRagEngine *engine, const char *c_content, const c_DocumentId c_document_id, const c_ScopeId c_scope_id)
{
    return true;
}

int32_t odai_generate_streaming_response(const char *c_query, const c_ScopeId c_scope_id,
                                         odai_stream_resp_callback_fn c_callback, void *c_user_data)
{
    try
    {
        if (!sdk_initialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return -1;
        }

        if (g_ragEngine == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "rag engine is not yet initalized");
            return -1;
        }

        if ((c_query == nullptr) || (c_scope_id == nullptr))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid query / scopeId is passed");
            return -1;
        }

        if (c_callback == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "empty callback passed");
            return -1;
        }

        string query(c_query);
        ScopeId scope_id(c_scope_id);
        int32_t total_tokens = g_ragEngine->generate_streaming_response(query, scope_id, c_callback, c_user_data);
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

bool odai_create_chat(const c_ChatId c_chat_id_in, const c_ChatConfig *c_chat_config, c_ChatId c_chat_id_out, size_t *chat_id_out_len)
{
    try
    {
        if (!sdk_initialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        ChatId chat_id;

        if (!is_sane(c_chat_config))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
            return false;
        }

        // if chat_id is null then we will generate and set the out, else we will use the given chat_id, given it's unique
        if (c_chat_id_in == nullptr)
        {
            chat_id = generate_chat_id();
            set_cstr_and_len(chat_id, c_chat_id_out, chat_id_out_len);
        }
        else
        {
            chat_id = c_chat_id_in;
            if (g_db->chat_id_exists(chat_id))
            {
                ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} already exists", c_chat_id_in);
                return false;
            }
        }

        if (!g_db->create_chat(chat_id, toCpp(*c_chat_config)))
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

bool odai_load_chat(const c_ChatId c_chat_id)
{
    try
    {
        if (!sdk_initialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if ((c_chat_id == nullptr))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
            return false;
        }

        ChatId chat_id(c_chat_id);

        if (!g_ragEngine->load_chat_session(chat_id))
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

bool odai_get_chat_history(const c_ChatId c_chat_id, c_ChatMessage **c_messages_out, size_t *messages_count)
{
    try
    {
        if (!sdk_initialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if ((c_chat_id == nullptr))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
            return false;
        }

        if (c_messages_out == nullptr || messages_count == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid output parameters passed");
            return false;
        }

        ChatId chat_id(c_chat_id);
        vector<ChatMessage> messages;

        if (!g_db->get_chat_history(chat_id, messages))
        {
            *c_messages_out = nullptr;
            *messages_count = 0;
            return false;
        }

        if (messages.empty())
        {
            *c_messages_out = nullptr;
            *messages_count = 0;
            return true;
        }

        // Allocate array of i_ChatMessage
        *messages_count = messages.size();
        *c_messages_out = static_cast<c_ChatMessage *>(malloc(sizeof(c_ChatMessage) * (*messages_count)));

        if (*c_messages_out == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "failed to allocate memory for messages");
            *messages_count = 0;
            return false;
        }

        // Convert each C++ message to C message
        for (size_t i = 0; i < messages.size(); ++i)
        {
            (*c_messages_out)[i] = toC(messages[i]);
        }

        return true;
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");

        if (c_messages_out != nullptr)
            *c_messages_out = nullptr;

        if (messages_count != nullptr)
            *messages_count = 0;
        return false;
    }
}

void odai_free_chat_messages(c_ChatMessage *c_messages, size_t count)
{
    if (c_messages == nullptr)
        return;

    try
    {
        // Free each message's dynamically allocated strings
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

        // Free the array itself
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
    try
    {
        // Sanity check: SDK initialization
        if (!sdk_initialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        // Sanity check: RAG engine initialization
        if (g_ragEngine == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "RAG engine is not yet initialized");
            return false;
        }

        if ((c_chat_id == nullptr))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
            return false;
        }

        if ((c_query == nullptr))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
            return false;
        }

        if ((c_scope_id == nullptr))
        {
            ODAI_LOG(ODAI_LOG_WARN, "Null / Empty scope_id passed: will be ignored if RAG is disabled");
            return -1;
        }

        if (callback == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid callback passed");
            return false;
        }

        ChatId chat_id(c_chat_id);
        string query(c_query);
        ScopeId scope_id(c_scope_id);

        // Call the RAG engine's generate_streaming_chat_response method
        // which will handle context retrieval and generation using cached context
        int32_t total_tokens = g_ragEngine->generate_streaming_chat_response(chat_id, query, scope_id, callback, user_data);

        if (total_tokens < 0)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming chat response for chat_id: {}", chat_id);
            return false;
        }

        ODAI_LOG(ODAI_LOG_INFO, "Successfully generated streaming chat response for chat_id: {} with {} tokens",
                 chat_id, total_tokens);

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught in odai_generate_streaming_chat_response: {}", e.what());
        return false;
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught in odai_generate_streaming_chat_response");
        return false;
    }
}