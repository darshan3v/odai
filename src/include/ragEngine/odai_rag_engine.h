#pragma once

#include "backendEngine/odai_backend_engine.h"
#include "db/odai_db.h"
#include "types/odai_types.h"
#include <unordered_map>

/// RAG (Retrieval-Augmented Generation) engine that combines embedding and language models for context-aware text generation.
/// Manages the initialization of models and generation of streaming responses using retrieved context.
class ODAIRagEngine
{
public:

    ODAIRagEngine(ODAIDb* db, ODAIBackendEngine* backendEngine);

    /// Registers a new model in the system with the given name and path.
    /// The model path is validated and a checksum is computed to ensure integrity.
    /// @param name The unique name to assign to the model.
    /// @param path The full file system path to the model file.
    /// @param type The type of the model.
    /// @return true if registration succeeded, false if name exists or file is invalid.
    bool register_model(const ModelName& name, const ModelPath& path, ModelType type);

    /// Updates the path for an existing model.
    /// Validates that the new file has the same checksum as the originally registered model.
    /// @param name The name of the model to update.
    /// @param path The new full file system path to the model file.
    /// @return true if update succeeded, false if validation fails or model not found.
    bool update_model_path(const ModelName& name, const ModelPath& path);

    /// Generates a streaming response for the given query.
    /// The response is streamed incrementally via the callback function.
    /// @param llm_model_config The Language Model and its config to be used for response generation
    /// @param query The input query/prompt to generate a response for
    /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
    /// @param callback Function called for each chunk of generated text. Can return false to cancel streaming.
    /// @param user_data User-provided data passed to the callback function
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_response(const LLMModelConfig& llm_model_config, const string &query, const SamplerConfig &sampler_config, odai_stream_resp_callback_fn callback, void *user_data);
    
    /// Loads a chat from the database to backend engine using the provided chat ID.
    /// Retrieves the chat configuration and loads the appropriate language model.
    /// If the chat uses RAG, the embedding model will be loaded.
    /// @param chat_id Unique identifier for the chat session to load
    /// @return true if chat session loaded successfully, false if chat_id not found or model loading failed
    bool load_chat_session(const ChatId &chat_id);

    /// Generates a streaming response for the given query for the given chat.
    /// Uses the previously loaded chat if cached, else will load chat and then generates a response.
    /// If RAG is enabled for the chat, retrieves relevant context from the knowledge base.
    /// @param chat_id Unique identifier for the chat session
    /// @param query The input query/message to generate a response for
    /// @param generator_config (Sampler, RAG settings, etc.)
    /// @param scope_id Scope identifier to filter documents during RAG retrieval (ignored if RAG is disabled)
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback function
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, const GeneratorConfig &generator_config,
                                            odai_stream_resp_callback_fn callback, void *user_data);

    /// Unloads the chat session from memory, freeing up resources.
    /// @param chat_id Unique identifier for the chat session
    /// @return true if unloaded successfully, false on error
    bool unload_chat_session(const ChatId &chat_id);

private:

    /// Resolves the file system path for a given model name using cache or database.
    /// @param modelName The name of the model.
    /// @param path Output parameter for the resolved path.
    /// @return true if found, false otherwise.
    bool resolve_model_path(const ModelName& modelName, ModelPath& path);

    /// Helper to ensure chat session is loaded into memory (backend engine context).
    /// If not loaded, it retrieves history from DB and loads it.
    /// @param chat_id Unique identifier for the chat session
    /// @param chat_config Configuration for the chat
    /// @return true if session is loaded (or was already loaded), false on error
    bool ensure_chat_session_loaded(const ChatId &chat_id, const ChatConfig &chat_config);


    ODAIDb* m_db = nullptr;
    ODAIBackendEngine* m_backendEngine = nullptr;

    unordered_map<string, string> m_modelPathCache;
};