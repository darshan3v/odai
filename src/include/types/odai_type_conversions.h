#include "odai_ctypes.h"
#include "odai_types.h"

#include<nlohmann/json.hpp>

// The assumption here is that c / c++ param is sane, and hence we don't do sanity checks here

/// Converts c_ModelType to C++ ModelType enum
/// @param c The C-style model type identifier
/// @return Corresponding C++ ModelType enum
ModelType toCpp(c_ModelType c);

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

/// Converts a C-style chunking configuration to C++ style.
/// @param c C-style chunking configuration to convert
/// @return C++ ChunkingConfig with the converted configuration
ChunkingConfig toCpp(const c_ChunkingConfig &c);

/// Converts a C-style semantic space configuration to C++ style.
/// @param c C-style semantic space configuration to convert
/// @return C++ SemanticSpaceConfig with the converted configuration
SemanticSpaceConfig toCpp(const c_SemanticSpaceConfig &c);

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

/// Converts a C++ EmbeddingModelConfig to C-style c_EmbeddingModelConfig.
/// Allocates memory for string fields that must be freed by the caller.
c_EmbeddingModelConfig toC(const EmbeddingModelConfig& cpp);

/// Converts a C++ ChunkingConfig to C-style c_ChunkingConfig.
c_ChunkingConfig toC(const ChunkingConfig& cpp);

/// Converts a C++ SemanticSpaceConfig to C-style c_SemanticSpaceConfig.
/// Allocates memory for string fields that must be freed by the caller.
c_SemanticSpaceConfig toC(const SemanticSpaceConfig& cpp);

/// Converts a C++ ChatMessage to C-style c_ChatMessage.
/// Allocates memory for content and message_metadata strings that must be freed by the caller.
/// @param cpp C++ ChatMessage to convert
/// @return C-style c_ChatMessage with allocated strings
c_ChatMessage toC(const ChatMessage& cpp);

// This creates to_json() and from_json() functions automatically.
/// Defines JSON serialization for LLMModelConfig.
/// Enables automatic conversion between LLMModelConfig and JSON using nlohmann/json.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LLMModelConfig, modelName)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EmbeddingModelConfig, modelName)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FixedSizeChunkingConfig, chunkSize, chunkOverlap)

void to_json(json& j, const ChunkingConfig& p);
void from_json(const json& j, ChunkingConfig& p);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SemanticSpaceConfig, name, embeddingModelConfig, chunkingConfig, dimensions)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ChatConfig, persistence, system_prompt, llmModelConfig)