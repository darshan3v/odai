#include "odai_ctypes.h"
#include "odai_types.h"

#include <nlohmann/json.hpp>

// The assumption here is that c / c++ param is sane, and hence we don't do
// sanity checks here

/// Converts c_ModelType to C++ ModelType enum
/// @param c The C-style model type identifier
/// @return Corresponding C++ ModelType enum
ModelType to_cpp(c_ModelType c);

/// Converts a C-style database configuration to C++ style.
/// Creates a new C++ DBConfig by copying the database type and path from the C
/// struct.
/// @param c C-style database configuration to convert
/// @return C++ DBConfig with the converted configuration
DBConfig to_cpp(const c_DbConfig& c);

/// Converts a C-style backend engine configuration to C++ style.
/// Creates a new C++ BackendEngineConfig by copying the engine type from the C
/// struct.
/// @param c C-style backend engine configuration to convert
/// @return C++ BackendEngineConfig with the converted configuration
BackendEngineConfig to_cpp(const c_BackendEngineConfig& c);

/// Converts a C-style embedding model configuration to C++ style.
/// Creates a new C++ EmbeddingModelConfig by copying the model path from the C
/// struct.
/// @param c C-style embedding model configuration to convert
/// @return C++ EmbeddingModelConfig with the converted configuration
EmbeddingModelConfig to_cpp(const c_EmbeddingModelConfig& c);

/// Converts a C-style language model configuration to C++ style.
/// Creates a new C++ LLMModelConfig by copying the model path from the C
/// struct.
/// @param c C-style language model configuration to convert
/// @return C++ LLMModelConfig with the converted configuration
LLMModelConfig to_cpp(const c_LlmModelConfig& c);

/// Converts a C-style chunking configuration to C++ style.
/// @param c C-style chunking configuration to convert
/// @return C++ ChunkingConfig with the converted configuration
ChunkingConfig to_cpp(const c_ChunkingConfig& c);

/// Converts a C-style semantic space configuration to C++ style.
/// @param c C-style semantic space configuration to convert
/// @return C++ SemanticSpaceConfig with the converted configuration
SemanticSpaceConfig to_cpp(const c_SemanticSpaceConfig& c);

/// Converts a C-style Retrieval configuration to C++ style.
/// @param c C-style Retrieval configuration to convert
/// @return C++ RetrievalConfig with the converted configuration
RetrievalConfig to_cpp(const c_RetrievalConfig& c);

/// Converts a C-style Sampler configuration to C++ style.
/// @param c C-style Sampler configuration to convert
/// @return C++ SamplerConfig with the converted configuration
SamplerConfig to_cpp(const c_SamplerConfig& c);

/// Converts a C-style Generator Rag configuration to C++ style.
/// @param c C-style Generator Rag configuration to convert
/// @return C++ GeneratorRagConfig with the converted configuration
GeneratorRagConfig to_cpp(const c_GeneratorRagConfig& source);

/// Converts a C-style Generator configuration to C++ style.
/// @param c C-style Generator configuration to convert
/// @return C++ GeneratorConfig with the converted configuration
GeneratorConfig to_cpp(const c_GeneratorConfig& source);

/// Converts a C-style chat configuration to C++ style.
/// Creates a new C++ ChatConfig by copying all fields from the C struct.
/// @param c C-style chat configuration to convert
/// @return C++ ChatConfig with the converted configuration
ChatConfig to_cpp(const c_ChatConfig& c);

/// Converts a C++ EmbeddingModelConfig to C-style c_EmbeddingModelConfig.
/// Allocates memory for string fields that must be freed by the caller.
c_EmbeddingModelConfig to_c(const EmbeddingModelConfig& cpp);

/// Converts a C++ ChunkingConfig to C-style c_ChunkingConfig.
c_ChunkingConfig to_c(const ChunkingConfig& cpp);

/// Converts a C++ SemanticSpaceConfig to C-style c_SemanticSpaceConfig.
/// Allocates memory for string fields that must be freed by the caller.
c_SemanticSpaceConfig to_c(const SemanticSpaceConfig& cpp);

/// Converts a C++ ChatMessage to C-style c_ChatMessage.
/// Allocates memory for content and message_metadata strings that must be freed
/// by the caller.
/// @param cpp C++ ChatMessage to convert
/// @return C-style c_ChatMessage with allocated strings
c_ChatMessage to_c(const ChatMessage& cpp);

// This creates to_json() and from_json() functions automatically.
/// Defines JSON serialization for LLMModelConfig.
/// Enables automatic conversion between LLMModelConfig and JSON using
/// nlohmann/json.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LLMModelConfig, m_modelName)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EmbeddingModelConfig, m_modelName)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FixedSizeChunkingConfig, m_chunkSize, m_chunkOverlap)

void to_json(json& j, const ChunkingConfig& p);
void from_json(const json& j, ChunkingConfig& p);

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SemanticSpaceConfig, m_name, m_embeddingModelConfig, m_chunkingConfig, m_dimensions)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ChatConfig, m_persistence, m_systemPrompt, m_llmModelConfig)