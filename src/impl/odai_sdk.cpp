#include "odai_sdk.h"
#include "ragEngine/odai_rag_engine.h"

#include "db/odai_sqlite_db.h"
#include "backendEngine/odai_llama_backend_engine.h"
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
    // Clean up if needed, though smart pointers in globals handle themselves mostly
    // But we might want to shut down explicitly if order matters
}

void ODAISdk::set_logger(odai_log_callback_fn callback, void *user_data)
{
    try
    {
        m_logger->odai_set_logger(callback, user_data);
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
        m_logger->odai_set_log_level(log_level);
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
    }
}

bool ODAISdk::initialize_sdk(const DBConfig& dbConfig, const BackendEngineConfig& backendConfig)
{
    try
    {
        if (!dbConfig.is_sane())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid db config passed");
            return false;
        }

        if (dbConfig.dbType == SQLITE_DB)
        {
            m_db = std::make_unique<ODAISqliteDb>(dbConfig);
        }

        if ((m_db.get() == nullptr) || (!m_db->initialize_db()))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize db");
            m_sdkInitialized = false;
            return false;
        }

        if (!backendConfig.is_sane())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid backend engine config passed");
            return false;
        }
        
        // Initialize the backend engine based on engineType
        if (backendConfig.engineType == LLAMA_BACKEND_ENGINE)
            m_backendEngine = std::make_unique<ODAILlamaEngine>(backendConfig);

        if ((m_backendEngine.get() == nullptr) || (!m_backendEngine->initialize_engine()))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize backend engine");
            m_sdkInitialized = false;
            return false;
        }

        // Initalize the RAGEngine
        m_ragEngine = make_unique<ODAIRagEngine>(m_db.get(),m_backendEngine.get());

        if ((m_ragEngine.get() == nullptr))
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

bool ODAISdk::add_document(const string& content, const DocumentId& documentId, const SemanticSpaceName& semanticSpaceName, const ScopeId& scopeId)
{
    try
    {
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        //ToDo : call rag engine to add document

        ODAI_LOG(ODAI_LOG_INFO, "Adding document to space: {}", semanticSpaceName);
        
        return true;
    }
    catch (...)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Exception caught");
        return false;
    }
}

int32_t ODAISdk::generate_streaming_response(const LLMModelConfig &llm_model_config, const string& query, 
                                          odai_stream_resp_callback_fn callback, void *userData)
{
    try
    {

        
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return -1;
        }
        
        if(!llm_model_config.is_sane())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid LLM Model Config passed");
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

        int32_t total_tokens = m_ragEngine->generate_streaming_response(llm_model_config, query, callback, userData);
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

bool ODAISdk::create_chat(const ChatId& chatIdIn, const ChatConfig& chatConfig, ChatId& chatIdOut)
{
    try
    {
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        ChatId chatId;

        if (!chatConfig.is_sane())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_config passed");
            return false;
        }

        // if chat_id is empty then we will generate and set the out, else we will use the given chat_id, given it's unique
        if (chatIdIn.empty())
        {
            chatId = generate_chat_id();
            chatIdOut = chatId;
        }
        else
        {
            chatId = chatIdIn;
            chatIdOut = chatId; // Set output to input in this case
            if (m_db->chat_id_exists(chatId))
            {
                ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} already exists", chatIdIn);
                return false;
            }
        }

        if (!m_db->create_chat(chatId, chatConfig))
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

bool ODAISdk::load_chat(const ChatId& chatId)
{
    try
    {
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if (chatId.empty())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
            return false;
        }

        if (!m_ragEngine->load_chat_session(chatId))
        {
            ODAI_LOG(ODAI_LOG_ERROR, "failed to load chat session, chat_id: {}", chatId);
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

bool ODAISdk::get_chat_history(const ChatId& chatId, vector<ChatMessage>& messages)
{
    try
    {
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if (chatId.empty())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
            return false;
        }

        if (!m_db->get_chat_history(chatId, messages))
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

bool ODAISdk::generate_streaming_chat_response(const ChatId& chatId, const string& query, const SemanticSpaceName& semanticSpaceName, const ScopeId& scopeId,
                                           odai_stream_resp_callback_fn callback, void *userData)
{
    try
    {
        // Sanity check: SDK initialization
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if (chatId.empty())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat_id passed");
            return false;
        }

        if (query.empty())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid query passed");
            return false;
        }

        if (scopeId.empty())
        {
            ODAI_LOG(ODAI_LOG_WARN, "Null / Empty scope_id passed: will be ignored if RAG is disabled");
            // Assuming this is just a warning and not an error unless strictly required
        }

        if (callback == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Invalid callback passed");
            return false;
        }

        // Call the RAG engine's generate_streaming_chat_response method
        int32_t total_tokens = m_ragEngine->generate_streaming_chat_response(chatId, query, semanticSpaceName, scopeId, callback, userData);

        if (total_tokens < 0)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to generate streaming chat response for chat_id: {}", chatId);
            return false;
        }

        ODAI_LOG(ODAI_LOG_INFO, "Successfully generated streaming chat response for chat_id: {} with {} tokens",
                 chatId, total_tokens);

        return true;
    }
    catch (const std::exception &e)
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

bool ODAISdk::unload_chat(const ChatId& chatId)
{
    try
    {
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if (chatId.empty())
        {
            ODAI_LOG(ODAI_LOG_ERROR, "invalid chat_id passed");
            return false;
        }

        return m_ragEngine->unload_chat_session(chatId);
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
        if (!m_sdkInitialized)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "SDK is not initialized");
            return false;
        }

        if(config.is_sane() == false)
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
        if (!m_sdkInitialized)
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
        if (!m_sdkInitialized)
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
        if (!m_sdkInitialized)
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
