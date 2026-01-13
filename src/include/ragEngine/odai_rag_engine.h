#pragma once

#include<memory>

#include "types/odai_types.h"

/// RAG (Retrieval-Augmented Generation) engine that combines embedding and language models for context-aware text generation.
/// Manages the initialization of models and generation of streaming responses using retrieved context.
class ODAIRagEngine
{
public:
    /// Initializes the RAG engine with the provided configuration.
    /// Loads both the embedding model and language model specified in the configuration.
    /// If initialization fails at any step, the function returns false and logs an error.
    /// @param config Configuration for RAG
    /// @return true if both models loaded successfully, false otherwise
    bool initialize(const RagConfig& config);

    /// Generates a streaming response for the given query using RAG context retrieval.
    /// Retrieves relevant context based on the scope ID and generates a response using the loaded language model.
    /// The response is streamed incrementally via the callback function.
    /// @param query The input query/prompt to generate a response for
    /// @param scopeId The scope identifier used to retrieve relevant context from the knowledge base
    /// @param callback Function called for each chunk of generated text. Can return false to cancel streaming.
    /// @param user_data User-provided data passed to the callback function
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_response(const string &query, const ScopeId &scopeId, odai_stream_resp_callback_fn callback, void *user_data);
    
    /// Loads a chat from the database to backend engine using the provided chat ID.
    /// Retrieves the chat configuration and loads the appropriate language model.
    /// If the chat uses RAG, the embedding model will be loaded if not already loaded.
    /// @param chat_id Unique identifier for the chat session to load
    /// @return true if chat session loaded successfully, false if chat_id not found or model loading failed
    bool load_chat_session(const ChatId &chat_id);

    /// Generates a streaming response for the given query for the given chat.
    /// Uses the previously loaded chat and generates a response.
    /// If RAG is enabled for the chat, retrieves relevant context from the knowledge base.
    /// @param chat_id Unique identifier for the chat session
    /// @param query The input query/message to generate a response for
    /// @param scope_id Scope identifier to filter documents during RAG retrieval (ignored if RAG is disabled)
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback function
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, const ScopeId &scope_id,
                                            odai_stream_resp_callback_fn callback, void *user_data);
};

/// Global RAG engine instance. Initialized when the RAG engine is set up.
inline unique_ptr<ODAIRagEngine> g_ragEngine = nullptr;