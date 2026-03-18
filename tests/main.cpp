#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/include/odai_public.h"

namespace fs = std::filesystem;

// --- Configuration Paths (Initialized in main) ---
static std::string TEST_BASE_PATH;
static std::string DB_PATH;
static std::string CACHE_PATH;

// Model Names
static const char* EMBEDDING_MODEL_NAME = "gemma-300m-embedding";
static const char* GEMMA3_MODEL_NAME = "gemma-3-4b-it";
static const char* GEMMA3N_MODEL_NAME = "gemma-3n-e2b-it";
static const char* QWEN_OMNI_MODEL_NAME = "qwen-2.5-omni";

// Model Paths
static std::string EMBEDDING_MODEL_PATH;
static std::string GEMMA3_LLM_PATH;
static std::string GEMMA3_MMPROJ_PATH;
static std::string GEMMA3N_LLM_PATH;
static std::string QWEN_OMNI_LLM_PATH;
static std::string QWEN_OMNI_MMPROJ_PATH;

// Data Paths
static std::string IMAGE_PATH;
static std::string AUDIO_PATH;

// Hyperparameters
static constexpr uint32_t MAX_TOKENS = 2048;
static constexpr float TEMPERATURE = 0.9F;
static constexpr uint32_t TOP_K = 40;
static constexpr uint32_t EMBEDDING_DIM = 384;
static constexpr uint32_t CHUNK_SIZE = 512;
static constexpr uint32_t CHUNK_OVERLAP = 50;

// RAG Parameters
static constexpr uint32_t RAG_TOP_K = 5;
static constexpr uint32_t RAG_FETCH_K = 20;
static constexpr float RAG_SCORE_THRESHOLD = 0.7F;
static constexpr uint32_t RAG_CONTEXT_WINDOW = 2;

static std::ofstream g_log;

// --- Callbacks ---
static void log_callback(OdaiLogLevel level, const char* msg, void* user_data)
{
  (void)level;
  (void)user_data;
  if (g_log.is_open())
  {
    g_log << msg << "\n";
  }
}

static bool stream_callback(const char* chunk, void* user_data)
{
  (void)user_data;
  std::cout << chunk << std::flush;
  return true;
}

static void print_semantic_spaces(const c_SemanticSpaceConfig* spaces, uint16_t count)
{
  if (spaces == nullptr)
  {
    return;
  }
  std::cout << "Found " << count << " spaces:\n";
  for (uint16_t i = 0; i < count; ++i)
  {
    const c_SemanticSpaceConfig& space = spaces[i];
    std::cout << " - Space: " << (space.m_name != nullptr ? space.m_name : "NULL") << "\n";
    std::cout << "   Model: "
              << (space.m_embeddingModelConfig.m_modelName != nullptr ? space.m_embeddingModelConfig.m_modelName
                                                                      : "NULL")
              << "\n";
    std::cout << "   Dims:  " << space.m_dimensions << "\n";
  }
}

static void print_chat_history(const c_ChatMessage* messages, uint16_t count)
{
  if (messages == nullptr)
  {
    return;
  }
  for (uint16_t i = 0; i < count; ++i)
  {
    const c_ChatMessage& message = messages[i];
    std::cout << "[" << static_cast<const char*>(message.m_role) << "]: ";
    for (size_t j = 0; j < message.m_contentItemsCount; ++j)
    {
      const c_InputItem& item = message.m_contentItems[j];
      if (item.m_type == ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER)
      {
        std::cout << static_cast<const char*>(item.m_data);
      }
      else if (item.m_type == ODAI_INPUT_ITEM_TYPE_FILE_PATH)
      {
        std::cout << "<File: " << static_cast<const char*>(item.m_data) << ">";
      }
    }
    std::cout << "\n";
  }
}

// --- Helpers ---
static bool init_sdk()
{
  g_log.open(TEST_BASE_PATH + "/odai.log");
  odai_set_log_level(ODAI_LOG_DEBUG);
  odai_set_logger(log_callback, nullptr);

  c_DbConfig db_conf = {SQLITE_DB, DB_PATH.c_str(), CACHE_PATH.c_str()};
  c_BackendEngineConfig backend_conf = {LLAMA_BACKEND_ENGINE};

  if (!odai_initialize_sdk(&db_conf, &backend_conf))
  {
    std::cerr << "Failed to initialize SDK\n";
    return false;
  }
  return true;
}

static bool test_registration()
{
  std::cout << "\n--- Testing Model Registration ---\n";

  // 1. Embedding Model
  c_ModelFileEntry embedding_entry = {"base_model_path", EMBEDDING_MODEL_PATH.c_str()};
  c_ModelFiles embedding_files = {ODAI_MODEL_TYPE_EMBEDDING, LLAMA_BACKEND_ENGINE, &embedding_entry, 1};

  if (odai_register_model_files(const_cast<char*>(EMBEDDING_MODEL_NAME), &embedding_files) != ODAI_SUCCESS)
  {
    std::cout << "Failed to register embedding model: " << EMBEDDING_MODEL_NAME << "\n";
  }
  else
  {
    std::cout << "Successfully registered embedding model: " << EMBEDDING_MODEL_NAME << "\n";
  }

  // 2. Gemma 3 Model
  std::array<c_ModelFileEntry, 2> gemma3_entries = {
      {{"base_model_path", GEMMA3_LLM_PATH.c_str()}, {"mmproj_model_path", GEMMA3_MMPROJ_PATH.c_str()}}};
  c_ModelFiles gemma3_files = {ODAI_MODEL_TYPE_LLM, LLAMA_BACKEND_ENGINE, gemma3_entries.data(), gemma3_entries.size()};

  if (odai_register_model_files(const_cast<char*>(GEMMA3_MODEL_NAME), &gemma3_files) != ODAI_SUCCESS)
  {
    std::cout << "Failed to register LLM model: " << GEMMA3_MODEL_NAME << "\n";
  }
  else
  {
    std::cout << "Successfully registered LLM model: " << GEMMA3_MODEL_NAME << "\n";
  }

  // 3. Gemma 3n Model
  c_ModelFileEntry gemma3n_entry = {"base_model_path", GEMMA3N_LLM_PATH.c_str()};
  c_ModelFiles gemma3n_files = {ODAI_MODEL_TYPE_LLM, LLAMA_BACKEND_ENGINE, &gemma3n_entry, 1};

  if (odai_register_model_files(const_cast<char*>(GEMMA3N_MODEL_NAME), &gemma3n_files) != ODAI_SUCCESS)
  {
    std::cout << "Failed to register LLM model: " << GEMMA3N_MODEL_NAME << "\n";
  }
  else
  {
    std::cout << "Successfully registered LLM model: " << GEMMA3N_MODEL_NAME << "\n";
  }

  // 4. Qwen Omni Model
  std::array<c_ModelFileEntry, 2> qwen_entries = {
      {{"base_model_path", QWEN_OMNI_LLM_PATH.c_str()}, {"mmproj_model_path", QWEN_OMNI_MMPROJ_PATH.c_str()}}};
  c_ModelFiles qwen_files = {ODAI_MODEL_TYPE_LLM, LLAMA_BACKEND_ENGINE, qwen_entries.data(), qwen_entries.size()};

  if (odai_register_model_files(const_cast<char*>(QWEN_OMNI_MODEL_NAME), &qwen_files) != ODAI_SUCCESS)
  {
    std::cout << "Failed to register LLM model: " << QWEN_OMNI_MODEL_NAME << "\n";
  }
  else
  {
    std::cout << "Successfully registered LLM model: " << QWEN_OMNI_MODEL_NAME << "\n";
  }

  // Test ALREADY_EXISTS behavior
  std::cout << "\n--- Testing Model Already Exists Constraint ---\n";
  c_OdaiResult duplicate_res = odai_register_model_files(const_cast<char*>(EMBEDDING_MODEL_NAME), &embedding_files);
  if (duplicate_res == ODAI_ALREADY_EXISTS)
  {
    std::cout << "Successfully detected duplicate model registration: " << EMBEDDING_MODEL_NAME << "\n";
  }
  else
  {
    std::cout << "Failed to return ODAI_ALREADY_EXISTS for duplicate model. Got: " << duplicate_res << "\n";
  }

  return true;
}

static bool test_semantic_spaces()
{
  std::cout << "\n--- Testing Semantic Spaces ---\n";

  c_EmbeddingModelConfig embedding_conf = {const_cast<char*>(EMBEDDING_MODEL_NAME)};
  c_ChunkingConfig chunk_config = {FIXED_SIZE_CHUNKING, {{CHUNK_SIZE, CHUNK_OVERLAP}}};

  const char* space_name = "test_space";
  c_SemanticSpaceConfig space_config = {const_cast<char*>(space_name), embedding_conf, chunk_config, EMBEDDING_DIM};

  if (odai_create_semantic_space(&space_config))
  {
    std::cout << "Semantic space '" << space_name << "' created.\n";
  }

  const char* space_name2 = "test_space2";
  c_SemanticSpaceConfig space_config2 = {const_cast<char*>(space_name2), embedding_conf, chunk_config, EMBEDDING_DIM};

  if (odai_create_semantic_space(&space_config2))
  {
    std::cout << "Semantic space '" << space_name2 << "' created.\n";
  }

  // List to verify creation
  c_SemanticSpaceConfig* spaces_list = nullptr;
  uint16_t spaces_count = 0;
  if (odai_list_semantic_spaces(&spaces_list, &spaces_count))
  {
    print_semantic_spaces(spaces_list, spaces_count);
    odai_free_semantic_spaces_list(spaces_list, spaces_count);
  }

  // Delete Space
  std::cout << "Deleting semantic space '" << space_name2 << "'...\n";
  if (odai_delete_semantic_space(const_cast<char*>(space_name2)))
  {
    std::cout << "Successfully deleted semantic space.\n";
  }
  else
  {
    std::cout << "Failed to delete semantic space.\n";
  }

  // Verify Deletion
  if (odai_list_semantic_spaces(&spaces_list, &spaces_count))
  {
    std::cout << "Spaces remaining after deletion: " << spaces_count << "\n";
    print_semantic_spaces(spaces_list, spaces_count);
    odai_free_semantic_spaces_list(spaces_list, spaces_count);
  }

  return true;
}

static bool test_streaming_text(const char* model_name)
{
  std::cout << "\n--- Testing Streaming Response (Text) using " << model_name << " ---\n";

  c_LlmModelConfig llm_conf = {const_cast<char*>(model_name)};
  c_SamplerConfig sampler_conf = {MAX_TOKENS, TEMPERATURE, TOP_K};

  const char* prompt = "What is the capital of France?";
  c_InputItem item = {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)prompt, strlen(prompt),
                      const_cast<char*>("text/plain")};

  std::cout << "Assistant: ";
  if (odai_generate_streaming_response(&llm_conf, &item, 1, &sampler_conf, stream_callback, nullptr) == -1)
  {
    std::cout << "Failed to generate streaming response\n";
    return false;
  }
  std::cout << "\n";

  return true;
}

static bool test_streaming_image(const char* model_name)
{
  std::cout << "\n--- Testing Streaming Response (Image) using " << model_name << " ---\n";

  c_LlmModelConfig llm_conf = {const_cast<char*>(model_name)};
  c_SamplerConfig sampler_conf = {MAX_TOKENS, TEMPERATURE, TOP_K};

  const char* prompt = "Describe this image";
  std::array<c_InputItem, 2> items = {{
      {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)prompt, strlen(prompt), const_cast<char*>("text/plain")},
      {ODAI_INPUT_ITEM_TYPE_FILE_PATH, (void*)IMAGE_PATH.c_str(), IMAGE_PATH.size(), const_cast<char*>("image/jpeg")},
  }};

  std::cout << "Assistant: ";
  if (odai_generate_streaming_response(&llm_conf, items.data(), items.size(), &sampler_conf, stream_callback,
                                       nullptr) == -1)
  {
    std::cout << "Failed to generate streaming response\n";
    return false;
  }
  std::cout << "\n";

  return true;
}

static bool test_streaming_audio(const char* model_name)
{
  std::cout << "\n--- Testing Streaming Response (Audio) using " << model_name << " ---\n";

  c_LlmModelConfig llm_conf = {const_cast<char*>(model_name)};
  c_SamplerConfig sampler_conf = {MAX_TOKENS, TEMPERATURE, TOP_K};

  const char* prompt = "Describe this audio briefly";
  std::array<c_InputItem, 2> items = {{
      {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)prompt, strlen(prompt), const_cast<char*>("text/plain")},
      {ODAI_INPUT_ITEM_TYPE_FILE_PATH, (void*)AUDIO_PATH.c_str(), AUDIO_PATH.size(), const_cast<char*>("audio/mpeg")},
  }};

  std::cout << "Assistant: ";
  if (odai_generate_streaming_response(&llm_conf, items.data(), items.size(), &sampler_conf, stream_callback,
                                       nullptr) == -1)
  {
    std::cout << "Failed to generate streaming response\n";
    return false;
  }
  std::cout << "\n";

  return true;
}

static bool test_chat_multimodal(const char* model_name)
{
  std::cout << "\n--- Testing Chat (Multimodal) using " << model_name << " ---\n";

  c_LlmModelConfig llm_conf = {const_cast<char*>(model_name)};
  c_ChatConfig chat_config = {true, "You are a helpful assistant.", llm_conf};
  c_ChatId chat_id = nullptr;

  if (!odai_create_chat(nullptr, &chat_config, &chat_id))
  {
    std::cerr << "Failed to create chat\n";
    return false;
  }

  c_SamplerConfig sampler_conf = {MAX_TOKENS, TEMPERATURE, TOP_K};
  c_RetrievalConfig retrieval_config = {RAG_TOP_K, RAG_FETCH_K,       RAG_SCORE_THRESHOLD, SEARCH_TYPE_VECTOR_ONLY,
                                        false,     RAG_CONTEXT_WINDOW};
  c_GeneratorRagConfig rag_config = {retrieval_config, const_cast<char*>("test_space"), const_cast<char*>("default")};
  c_GeneratorConfig gen_config = {sampler_conf, RAG_MODE_DYNAMIC, &rag_config};

  // Turn 1: Image prompt
  const char* p1_text = "What is in this image?";
  std::array<c_InputItem, 2> prompt1 = {{
      {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)p1_text, strlen(p1_text), const_cast<char*>("text/plain")},
      {ODAI_INPUT_ITEM_TYPE_FILE_PATH, (void*)IMAGE_PATH.c_str(), IMAGE_PATH.size(), const_cast<char*>("image/jpeg")},
  }};

  std::cout << "User: " << p1_text << "\nAssistant: ";
  if (odai_generate_streaming_chat_response(chat_id, prompt1.data(), prompt1.size(), &gen_config, stream_callback,
                                            nullptr) == -1)
  {
    std::cerr << "Failed to generate streaming chat response\n";
    return false;
  }
  std::cout << "\n";

  // Turn 2: Audio prompt
  const char* p2_text = "Describe this audio briefly";
  std::array<c_InputItem, 2> prompt2 = {{
      {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)p2_text, strlen(p2_text), const_cast<char*>("text/plain")},
      {ODAI_INPUT_ITEM_TYPE_FILE_PATH, (void*)AUDIO_PATH.c_str(), AUDIO_PATH.size(), const_cast<char*>("audio/mpeg")},
  }};

  std::cout << "\nUser: " << p2_text << "\nAssistant: ";
  if (odai_generate_streaming_chat_response(chat_id, prompt2.data(), prompt2.size(), &gen_config, stream_callback,
                                            nullptr) == -1)
  {
    std::cerr << "Failed to generate streaming chat response\n";
    return false;
  }
  std::cout << "\n";

  // Turn 3: Follow-up joke
  const char* p3_text = "Tell a joke based on our previous conversation";
  c_InputItem prompt3 = {ODAI_INPUT_ITEM_TYPE_MEMORY_BUFFER, (void*)p3_text, strlen(p3_text),
                         const_cast<char*>("text/plain")};

  std::cout << "\nUser: " << p3_text << "\nAssistant: ";
  if (odai_generate_streaming_chat_response(chat_id, &prompt3, 1, &gen_config, stream_callback, nullptr) == -1)
  {
    std::cerr << "Failed to generate streaming chat response\n";
    return false;
  }
  std::cout << "\n";

  // History
  std::cout << "\n--- Printing Chat History ---\n";
  c_ChatMessage* messages = nullptr;
  uint16_t count = 0;
  if (odai_get_chat_history(chat_id, &messages, &count))
  {
    print_chat_history(messages, count);
    odai_free_chat_messages(messages, count);
  }

  odai_free_chat_id(chat_id);
  return true;
}

static void cleanup()
{
  std::cout << "\n--- Cleanup ---\n";
  try
  {
    if (fs::exists(DB_PATH))
    {
      fs::remove(DB_PATH);
    }
    if (fs::exists(CACHE_PATH))
    {
      fs::remove_all(CACHE_PATH);
    }
    std::cout << "Test artifacts removed.\n";
  }
  catch (const std::exception& e)
  {
    std::cerr << "Cleanup failed: " << e.what() << "\n";
  }
}

int main(int argc, char** argv)
{
  if (argc > 0)
  {
    TEST_BASE_PATH = fs::absolute(fs::path(argv[0]).parent_path()).string();
  }
  else
  {
    TEST_BASE_PATH = fs::current_path().string();
  }

  // Initialize dependent paths
  DB_PATH = TEST_BASE_PATH + "/odaisample.db";
  CACHE_PATH = TEST_BASE_PATH + "/cache";

  EMBEDDING_MODEL_PATH = TEST_BASE_PATH + "/models/embeddings/embeddinggemma-300M-Q8_0.gguf";
  GEMMA3_LLM_PATH = TEST_BASE_PATH + "/models/llms/gemma3/gemma-3-4b-it-Q4_K_M.gguf";
  GEMMA3_MMPROJ_PATH = TEST_BASE_PATH + "/models/llms/gemma3/gemma3-mmproj-model-f16.gguf";
  GEMMA3N_LLM_PATH = TEST_BASE_PATH + "/models/llms/gemma3n/gemma-3n-E2B-it-Q4_0.gguf";
  QWEN_OMNI_LLM_PATH = TEST_BASE_PATH + "/models/llms/qwen_2.5_omni/Qwen2.5-Omni-3B-Q4_K_M.gguf";
  QWEN_OMNI_MMPROJ_PATH = TEST_BASE_PATH + "/models/llms/qwen_2.5_omni/mmproj-Qwen2.5-Omni-3B-Q8_0.gguf";

  IMAGE_PATH = TEST_BASE_PATH + "/data/images/sample_chamaleon.jpg";
  AUDIO_PATH = TEST_BASE_PATH + "/data/audio/Echoes_of_Unseen_Light.mp3";

  // Easy to toggle tests
  bool run_streaming = true;
  bool run_chat = false;
  bool do_cleanup = true;

  if (do_cleanup)
  {
    cleanup();
  }

  if (!init_sdk())
  {
    return 1;
  }

  test_registration();
  test_semantic_spaces();

  if (run_streaming)
  {
    test_streaming_text(GEMMA3N_MODEL_NAME);
    test_streaming_image(GEMMA3_MODEL_NAME);
    test_streaming_audio(QWEN_OMNI_MODEL_NAME);
  }

  if (run_chat)
  {
    test_chat_multimodal(QWEN_OMNI_MODEL_NAME);
  }

  // No cleanup at the end, we do it at the start

  std::cout << "\nAll tests completed.\n";
  return 0;
}