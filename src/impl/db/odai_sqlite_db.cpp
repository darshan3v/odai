#include "db/odai_sqlite_db.h"
#include "odai_sdk.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "types/odai_type_conversions.h"

#include "utils/odai_helpers.h"
#include "xxhash.h"
#include <fstream>

using namespace nlohmann;

// SQLiteCpp uses try catch handling heavily so we use them here a lot

using namespace std;

extern "C" int sqlite3_vec_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* p_api);

OdaiSqliteDb::OdaiSqliteDb(const DBConfig& db_config, const string& cache_dir_path)
    : m_dbPath(db_config.m_dbPath), m_cacheDirPath(cache_dir_path)
{
}

bool OdaiSqliteDb::register_vec_extension()
{
  try
  {
    // because of the way sqlite3 is built, can't directly call sqlite3_vec_init, if you call it will crash
    int64_t ret_code = sqlite3_auto_extension((void (*)(void))sqlite3_vec_init);

    if (ret_code != SQLITE_OK)
    {
      const char* error_msg = sqlite3_errstr(ret_code);
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to register sqlite vec extension, code: {}, error : {}", ret_code, error_msg);
      return false;
    }

    ODAI_LOG(ODAI_LOG_INFO, "sqlite-vec extension registered successfully");

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "DB Init Error: {}", e.what());
    return false;
  }
}

bool OdaiSqliteDb::initialize_db()
{
  try
  {
    // Register vec extension first
    if (!register_vec_extension())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to register sqlite-vec extension");
      return false;
    }

    bool initialize_schema = false;

    // Check if DB file exists
    if (!filesystem::exists(m_dbPath))
    {
      initialize_schema = true;
      ODAI_LOG(ODAI_LOG_INFO, "Database file does not exist. It will be created at {}", m_dbPath);
    }

    // create db object only after registering sqlite-vec extension
    m_db = std::make_unique<SQLite::Database>(m_dbPath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

    ODAI_LOG(ODAI_LOG_INFO, "Opened / created database successfully at {}", m_dbPath);

    if (initialize_schema)
    {
      m_db->exec(db_schema);
      ODAI_LOG(ODAI_LOG_INFO, "initialized db with schema");
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize DB : {} Error: {}", m_dbPath, e.what());
    return false;
  }
}

bool OdaiSqliteDb::begin_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    m_transactionDepth++;
    if (m_transactionDepth == 1)
    {
      // Start the physical transaction
      m_transaction = std::make_unique<SQLite::Transaction>(*m_db);
    }
    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to begin transaction: {}", e.what());
    if (m_transactionDepth == 1)
    {
      m_transactionDepth = 0;
      m_transaction.reset();
    }
    else
    {
      m_transactionDepth--;
    }
    return false;
  }
}

bool OdaiSqliteDb::commit_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    if (m_transactionDepth > 0)
    {
      m_transactionDepth--;
      if (m_transactionDepth == 0)
      {
        // Commit the physical transaction
        if (m_transaction)
        {
          m_transaction->commit();
          m_transaction.reset();
        }
      }
      return true;
    }

    ODAI_LOG(ODAI_LOG_WARN, "commit_transaction called with no active transaction");
    return false;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction: {}", e.what());
    // If commit fails, we leave the transaction object (destructor will rollback if reset/destroyed)
    return false;
  }
}

bool OdaiSqliteDb::rollback_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    // Regardless of depth, we roll back everything
    // Destroying the Transaction object safely rolls it back if not committed
    m_transaction.reset();
    m_transactionDepth = 0;
    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to rollback transaction: {}", e.what());
    m_transactionDepth = 0;
    return false;
  }
}

bool OdaiSqliteDb::register_model_files(const ModelName& name, const ModelFiles& model_file_details,
                                        const string& checksums)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    if (checksums.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Empty checksums passed for model: {}", name);
      return false;
    }

    string type_str;
    if (model_file_details.m_modelType == ModelType::LLM)
    {
      type_str = "LLM";
    }
    else if (model_file_details.m_modelType == ModelType::EMBEDDING)
    {
      type_str = "EMBEDDING";
    }
    else
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed");
      return false;
    }

    json j = model_file_details;
    string model_file_details_json = j.dump();

    SQLite::Statement insert(*m_db, "INSERT INTO models (name, file_details, checksums, type) VALUES (:name, "
                                    "jsonb(:file_details), jsonb(:checksums), :type)");
    insert.bind(":name", name);
    insert.bind(":file_details", model_file_details_json);
    insert.bind(":checksums", checksums);
    insert.bind(":type", type_str);

    insert.exec();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to register model: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::get_model_files(const ModelName& name, ModelFiles& model_file_details)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    SQLite::Statement query(*m_db, "SELECT json(file_details) as file_details FROM models WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (query.executeStep())
    {
      SQLite::Column file_details_col = query.getColumn("file_details");
      json file_details_json = json::parse(file_details_col.getString());
      model_file_details = file_details_json.get<ModelFiles>();
      return true;
    }

    return false;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get model file details: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::get_model_checksums(const ModelName& name, string& checksums)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    SQLite::Statement query(*m_db, "SELECT json(checksums) as checksums FROM models WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (query.executeStep())
    {
      checksums = query.getColumn("checksums").getString();
      return true;
    }

    return false;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get model checksums: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::update_model_files(const ModelName& name, const ModelFiles& new_model_file_details,
                                      const string& new_checksums)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    json j = new_model_file_details;
    string new_model_file_details_json = j.dump();

    SQLite::Statement update(
        *m_db,
        "UPDATE models SET file_details = jsonb(:file_details), checksums = jsonb(:checksums) WHERE name = :name");
    update.bind(":file_details", new_model_file_details_json);
    update.bind(":checksums", new_checksums);
    update.bind(":name", name);

    update.exec();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to update file details for model: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::create_semantic_space(const SemanticSpaceConfig& config)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    if (!config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid semantic space config passed");
      return false;
    }

    json j = config;
    string config_json = j.dump();

    SQLite::Statement insert(*m_db, "INSERT INTO semantic_spaces (name, config) VALUES (:name, jsonb(:config))");
    insert.bind(":name", config.m_name);
    insert.bind(":config", config_json);

    insert.exec();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create semantic space: {}, Error: {}", config.m_name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    SQLite::Statement query(*m_db, "SELECT json(config) as config FROM semantic_spaces WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (!query.executeStep())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Semantic space not found: {}", name);
      return false;
    }

    SQLite::Column config_col = query.getColumn("config");
    json config_json = json::parse(config_col.getString());
    config = config_json.get<SemanticSpaceConfig>();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get semantic space config: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::list_semantic_spaces(vector<SemanticSpaceConfig>& spaces)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    spaces.clear();

    SQLite::Statement query(*m_db, "SELECT json(config) as config FROM semantic_spaces ORDER BY name");

    while (query.executeStep())
    {
      SQLite::Column config_col = query.getColumn("config");
      json config_json = json::parse(config_col.getString());
      spaces.push_back(config_json.get<SemanticSpaceConfig>());
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to list semantic spaces, Error: {}", e.what());
    return false;
  }
}

bool OdaiSqliteDb::delete_semantic_space(const SemanticSpaceName& name)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    SQLite::Statement query(*m_db, "DELETE FROM semantic_spaces WHERE name = :name");
    query.bind(":name", name);

    query.exec();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to delete semantic space: {}, Error: {}", name, e.what());
    return false;
  }
}

bool OdaiSqliteDb::chat_id_exists(const ChatId& chat_id)
{
  try
  {
    // "SELECT 1" is enough. We limit to 1 so the DB stops searching immediately.
    SQLite::Statement select_chat(*m_db, "SELECT 1 FROM chats WHERE chat_id = :chat_id LIMIT 1");

    // Bind the parameter, use named parameters instead of index based to avoid unexpected behaviour due to changes in
    // future
    select_chat.bind(":chat_id", chat_id);
    select_chat.exec();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Check Exists Error: {}", e.what());
    return false;
  }
}

bool OdaiSqliteDb::create_chat(const ChatId& chat_id, const ChatConfig& chat_config)
{
  try
  {

    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    if (!chat_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat config passed");
      return false;
    }

    json j = chat_config;
    string chat_config_json = j.dump();

    begin_transaction();

    SQLite::Statement insert_chat(*m_db,
                                  "INSERT INTO chats (chat_id, chat_config) VALUES (:chat_id, jsonb(:chat_config))");

    insert_chat.bind(":chat_id", chat_id);
    insert_chat.bind(":chat_config", chat_config_json);
    insert_chat.exec();

    // Insert system prompt as initial message
    ChatMessage system_msg;
    system_msg.m_role = "system";
    system_msg.m_contentItems.push_back(
        {InputItemType::MEMORY_BUFFER,
         vector<uint8_t>(chat_config.m_systemPrompt.begin(), chat_config.m_systemPrompt.end()), "text/plain"});
    system_msg.m_messageMetadata = {};
    insert_chat_messages(chat_id, {system_msg});

    commit_transaction();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create chat session Error: {}", e.what());
    return false;
  }
}

bool OdaiSqliteDb::get_chat_config(const ChatId& chat_id, ChatConfig& chat_config)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    // always use named fields to extract data instead of index
    SQLite::Statement query(
        *m_db, "SELECT title, json(chat_config) as chat_config FROM chats WHERE chat_id = :chat_id LIMIT 1");

    query.bind(":chat_id", chat_id);

    if (!query.executeStep())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} does not exist", chat_id);
      return false;
    }

    SQLite::Column chat_config_col = query.getColumn("chat_config");

    if (chat_config_col.isNull())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "chat_config is null for chat_id {}", chat_id);
      return false;
    }

    json chat_config_json = json::parse(chat_config_col.getString());

    chat_config = chat_config_json.get<ChatConfig>();

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to Load Chat, Chat Id : {}, Error: {}", chat_id, e.what());
    return false;
  }
}

bool OdaiSqliteDb::get_chat_history(const ChatId& chat_id, vector<ChatMessage>& messages)
{
  try
  {

    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    messages.clear();

    SQLite::Statement query(*m_db, "SELECT role, content, json(message_metadata) as message_metadata, created_at "
                                   "FROM chat_messages "
                                   "WHERE chat_id = :chat_id "
                                   "ORDER BY sequence_index");

    query.bind(":chat_id", chat_id);

    bool has_results = false;
    while (query.executeStep())
    {
      has_results = true;

      ChatMessage msg;

      SQLite::Column role_col = query.getColumn("role");
      SQLite::Column content_col = query.getColumn("content");
      SQLite::Column metadata_col = query.getColumn("message_metadata");
      SQLite::Column created_at_col = query.getColumn("created_at");

      msg.m_role = role_col.getString();

      json content_json = json::parse(content_col.getString());
      msg.m_contentItems = content_json.get<vector<InputItem>>();

      // Handle NULL message_metadata by defaulting to empty JSON object
      if (metadata_col.isNull())
      {
        msg.m_messageMetadata = {};
      }
      else
      {
        msg.m_messageMetadata = json::parse(metadata_col.getString());
      }

      // Cast to uint64_t since Unix timestamps are always non-negative
      msg.m_createdAt = static_cast<uint64_t>(created_at_col.getInt64());

      messages.push_back(msg);
    }

    if (!has_results)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} does not exist or has no messages", chat_id);
      return false;
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get chat history, Chat Id : {}, Error: {}", chat_id, e.what());
    return false;
  }
}

bool OdaiSqliteDb::process_and_cache_media_item(InputItem& item)
{
  if (item.m_type == InputItemType::PROCESSED_DATA)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Please give the raw media data and not decoded media data");
    return false;
  }

  MediaType media_type = get_media_type_from_mime(item.m_mimeType);

  if (media_type == MediaType::AUDIO || media_type == MediaType::IMAGE)
  {
    std::string absolute_path;

    if (item.m_data.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Empty data passed for media item");
      return false;
    }

    // Let's compute xxhash
    std::string hash_xxhash;
    if (item.m_type == InputItemType::FILE_PATH)
    {
      std::string file_path(item.m_data.begin(), item.m_data.end());
      hash_xxhash = calculate_file_checksum(file_path);
    }
    else
    {
      hash_xxhash = calculate_data_checksum(item.m_data);
    }

    if (hash_xxhash.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksum for media item");
      return false;
    }

    SQLite::Statement query_cache(*m_db, "SELECT absolute_path FROM media_cache WHERE hash_xxhash = ? LIMIT 1");
    query_cache.bind(1, hash_xxhash);

    if (query_cache.executeStep())
    {
      absolute_path = query_cache.getColumn(0).getString();
    }
    else
    {
      // Write to disk
      std::string ext = ".bin";
      if (media_type == MediaType::AUDIO)
      {
        ext = ".audio";
      }
      else if (media_type == MediaType::IMAGE)
      {
        ext = ".img";
      }

      std::string file_name = hash_xxhash + ext;
      std::filesystem::path full_path = std::filesystem::path(m_cacheDirPath) / file_name;
      absolute_path = full_path.string();
      std::ofstream out_file(absolute_path, std::ios::binary);

      if (!out_file.is_open())
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to open file for writing: {}", absolute_path);
        return false;
      }

      out_file.write(reinterpret_cast<const char*>(item.m_data.data()), item.m_data.size());
      out_file.close();

      SQLite::Statement insert_cache(
          *m_db, "INSERT INTO media_cache (id, hash_xxhash, mime_type, absolute_path, file_name) VALUES (?, "
                 "?, ?, ?, ?)");
      insert_cache.bind(1, "mc_" + hash_xxhash);
      insert_cache.bind(2, hash_xxhash);
      insert_cache.bind(3, item.m_mimeType);
      insert_cache.bind(4, absolute_path);
      insert_cache.bind(5, file_name);
      insert_cache.exec();
    }

    // Convert to FILE type and replace data with absolute path
    item.m_type = InputItemType::FILE_PATH;
    item.m_data.clear();
    item.m_data.insert(item.m_data.end(), absolute_path.begin(), absolute_path.end());
  }

  return true;
}

bool OdaiSqliteDb::insert_chat_messages(const ChatId& chat_id, const vector<ChatMessage>& messages)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return false;
    }

    if (messages.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "no messages passed to insert for chat_id: {}", chat_id);
      return true;
    }

    // Start transaction (supports nesting)
    begin_transaction();

    try
    {
      // Prepare statement once, reuse for all messages
      SQLite::Statement insert_message(
          *m_db, "INSERT INTO chat_messages (chat_id, role, content, message_metadata, sequence_index) "
                 "VALUES (:chat_id, :role, :content, jsonb(:message_metadata), "
                 "COALESCE((SELECT MAX(sequence_index) + 1 FROM chat_messages WHERE chat_id = :chat_id), 0))");

      // IMPORTANT: This function must always receive the original (true) data types like
      // AUDIO_FILE or IMAGE_BUFFER from the user prompt rather than internal decoded data
      // (like AUDIO_PCM). The deduplication and caching mechanisms rely on the raw/compressed
      // data formats to work efficiently.

      // Ensure cache directory exists if not empty
      if (!m_cacheDirPath.empty() && !std::filesystem::exists(m_cacheDirPath))
      {
        if (!std::filesystem::create_directories(m_cacheDirPath))
        {
          ODAI_LOG(ODAI_LOG_ERROR, "Failed to create cache directory: {}", m_cacheDirPath);
          return false;
        }
      }

      for (auto msg : messages)
      {
        for (auto& item : msg.m_contentItems)
        {
          if (!process_and_cache_media_item(item))
          {
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to process and cache media item");
            return false;
          }
        }

        insert_message.bind(":chat_id", chat_id);
        insert_message.bind(":role", msg.m_role);
        json content_json = msg.m_contentItems;
        insert_message.bind(":content", content_json.dump());
        insert_message.bind(":message_metadata", msg.m_messageMetadata.dump());
        insert_message.exec();
        insert_message.reset(); // Reset for next iteration
        insert_message.clearBindings();
      }

      commit_transaction();
    }
    catch (...)
    {
      rollback_transaction();
      throw; // Re-throw to be caught by outer catch
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to insert chat messages Error: {}", e.what());
    return false;
  }
}

void OdaiSqliteDb::close()
{
  try
  {
    if (m_db != nullptr)
    {
      m_db.reset();
      ODAI_LOG(ODAI_LOG_INFO, "Database connection closed successfully");
    }
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Error closing database: {}", e.what());
  }
}

OdaiSqliteDb::~OdaiSqliteDb() {}
