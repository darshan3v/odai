#include "odai_ctypes.h"
#include "odai_types.h"

#include<nlohmann/json.hpp>

/// Converts a C-style database configuration to C++ style.
/// Creates a new C++ DBConfig by copying the database type and path from the C struct.
/// @param c C-style database configuration to convert
/// @return C++ DBConfig with the converted configuration
DBConfig toCpp(const c_DBConfig &c);

/// Converts a C-style backend engine configuration to C++ style.
/// Creates a new C++ BackendEngineConfig by copying the engine type from the C struct.
/// @param c C-style backend engine configuration to convert
/// @return C++ BackendEngineConfig with the converted configuration
BackendEngineConfig toCpp(const c_BackendEngineConfig &c);

/// Converts a C-style embedding model configuration to C++ style.
/// Creates a new C++ EmbeddingModelConfig by copying the model path from the C struct.
/// @param c C-style embedding model configuration to convert
/// @return C++ EmbeddingModelConfig with the converted configuration
EmbeddingModelConfig toCpp(const c_EmbeddingModelConfig &c);

/// Converts a C-style language model configuration to C++ style.
/// Creates a new C++ LLMModelConfig by copying the model path from the C struct.
/// @param c C-style language model configuration to convert
/// @return C++ LLMModelConfig with the converted configuration
LLMModelConfig toCpp(const c_LLMModelConfig &c);

/// Converts a C-style RAG configuration to C++ style.
/// Creates a new C++ RagConfig by converting both embedding and LLM model configurations.
/// @param c C-style RAG configuration to convert
/// @return C++ RagConfig with the converted configuration
RagConfig toCpp(const c_RagConfig &c);

/// Converts a C-style chat configuration to C++ style.
/// Creates a new C++ ChatConfig by copying all fields from the C struct.
/// @param c C-style chat configuration to convert
/// @return C++ ChatConfig with the converted configuration
ChatConfig toCpp(const c_ChatConfig &c);

/// Converts a C++ ChatMessage to C-style c_ChatMessage.
/// Allocates memory for content and message_metadata strings that must be freed by the caller.
/// @param cpp C++ ChatMessage to convert
/// @return C-style c_ChatMessage with allocated strings
c_ChatMessage toC(const ChatMessage& cpp);

// This creates to_json() and from_json() functions automatically.

/// Defines JSON serialization for LLMModelConfig.
/// Enables automatic conversion between LLMModelConfig and JSON using nlohmann/json.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LLMModelConfig, modelPath)

/// Defines JSON serialization for ChatConfig.
/// Enables automatic conversion between ChatConfig and JSON using nlohmann/json.
/// Serializes persistence, system_prompt, and llmModelConfig fields.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ChatConfig, persistence, system_prompt, llmModelConfig)