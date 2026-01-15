#include "backendEngine/odai_backend_engine.h"
#include "odai_sdk.h"

#include "ragEngine/odai_rag_engine.h"
#include "backendEngine/odai_llama_backend_engine.h"
#include "db/odai_sqlite_db.h"
#include "types/odai_ctypes.h"

// Implementation for initializing the RAG engine with the provided configuration
bool ODAIRagEngine::initialize(const RagConfig &config, ODAIDb* db, ODAIBackendEngine* backendEngine)
{
    this->m_db = db;
    this->m_backendEngine = backendEngine;

    if (!config.is_sane())
    {
        ODAI_LOG(ODAI_LOG_ERROR, "invalid rag config passed");
        return false;
    }

    if (!m_backendEngine->load_embedding_model(config.embeddingModelConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed initialize RagEngine because failed to load Embedding Model");
        return false;
    }

    if (!m_backendEngine->load_language_model(config.llmModelConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed initialize RagEngine because failed to load Language Model");
        return false;
    }

    ODAI_LOG(ODAI_LOG_INFO, "RAG Engine successfully initialized");
    return true;
}

int32_t ODAIRagEngine::generate_streaming_response(const string &query, odai_stream_resp_callback_fn callback, void *user_data)
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

    return m_backendEngine->generate_streaming_response(query, callback, user_data);
}

bool ODAIRagEngine::load_chat_session(const ChatId &chat_id)
{
    ChatConfig chat_config;

    if (!m_db->get_chat_config(chat_id, chat_config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat config, chat_id: {}", chat_id);
        return false;
    }

    if (chat_config.use_rag)
    {
        // ToDo -> Load Embedding model if not already loaded
    }

    if (!m_backendEngine->load_language_model(chat_config.llmModelConfig))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat {}, error : failed to load language Model", chat_id);
        return false;
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

int32_t ODAIRagEngine::generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, const ScopeId &scope_id,
                                                        odai_stream_resp_callback_fn callback, void *user_data)
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
    if (chat_config.use_rag && scope_id.empty())
    {
        ODAI_LOG(ODAI_LOG_ERROR, "RAG is enabled for chat_id: {} but scope_id is empty", chat_id);
        return -1;
    }
    
    // If RAG is enabled and scope_id is provided, retrieve and combine context
    if (chat_config.use_rag && !scope_id.empty())
    {
        // ToDo -> Implement context retrieval from knowledge base based on scope_id
        // string context = retrieve_context_from_knowledge_base(scope_id, prompt);
        ODAI_LOG(ODAI_LOG_DEBUG, "RAG is enabled for chat_id: {} with scope_id: {}", chat_id, scope_id);
    }

    if (!this->ensure_chat_session_loaded(chat_id, chat_config))
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to ensure chat session is loaded for chat_id: {}", chat_id);
        return -1;
    }

    // final_prompt = combine_context_and_query(context, prompt);
    string final_prompt = prompt; // Placeholder until context retrieval is implemented

    StreamingBufferContext buffer_ctx;
    buffer_ctx.user_callback = callback;
    buffer_ctx.user_data = user_data;
    buffer_ctx.buffered_response = "";

    // Internal callback that buffers output and forwards to user callback
    auto internal_callback = [](const char *token, void *user_data) -> bool
    {
        if (token == nullptr || user_data == nullptr)
            return false;

        StreamingBufferContext *ctx = static_cast<StreamingBufferContext *>(user_data);

        // Buffer the token
        ctx->buffered_response += string(token);

        // Forward to user callback for streaming
        return ctx->user_callback(token, ctx->user_data);
    };

    // Generate streaming response with internal buffering callback
    int32_t total_tokens = m_backendEngine->generate_streaming_chat_response(chat_id, final_prompt, internal_callback, &buffer_ctx);

    if (total_tokens < 0)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming response for chat_id: {}", chat_id);
        return -1;
    }

    // Prepare messages to save
    vector<ChatMessage> messages_to_save;

    ChatMessage user_msg;
    user_msg.role = "user";
    user_msg.content = prompt;
    user_msg.message_metadata = json::object();
    messages_to_save.push_back(user_msg);

    // ToDo in message_metadata add citations if any from RAG context

    ChatMessage assistant_msg;
    assistant_msg.role = "assistant";
    assistant_msg.content = buffer_ctx.buffered_response;
    assistant_msg.message_metadata = json::object();
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

bool ODAIRagEngine::unload_chat_session(const ChatId &chat_id)
{
    if (this->m_backendEngine == nullptr)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Backend engine is null");
        return false;
    }
    return this->m_backendEngine->unload_chat_context(chat_id);
}

bool ODAIRagEngine::ensure_chat_session_loaded(const ChatId &chat_id, const ChatConfig &chat_config)
{
    // 1. Ensure the correct language model is loaded
    // This call handles checking if the model is already loaded (fast path),
    // and if not, it loads it and CLEARS existing contexts (slow path).
    if (!m_backendEngine->load_language_model(chat_config.llmModelConfig))
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

    // 3. If not loaded, we need to load it from history
    if (chat_config.use_rag)
    {
        // ToDo -> Load Embedding model if not already loaded
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