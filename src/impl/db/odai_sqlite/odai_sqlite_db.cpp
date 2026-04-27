#include "db/odai_sqlite/odai_sqlite_db.h"
#include "odai_sdk.h"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include "types/odai_type_conversions.h"

#include "types/odai_types.h"
#include "utils/odai_helpers.h"
#include "xxhash.h"
#include <fstream>

// SQLiteCpp uses try catch handling heavily so we use them here a lot

extern "C" int sqlite3_vec_init(sqlite3* db, char** pz_err_msg, const sqlite3_api_routines* p_api);

namespace
{
OdaiResult<std::string> to_model_type_db_value(ModelType model_type)
{
  if (model_type == ModelType::LLM)
  {
    return std::string{"LLM"};
  }
  if (model_type == ModelType::EMBEDDING)
  {
    return std::string{"EMBEDDING"};
  }

  return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
}
} // namespace

OdaiSqliteDb::OdaiSqliteDb(const DBConfig& db_config) : IOdaiDb(db_config)
{
  if (db_config.m_dbType != SQLITE_DB)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unsupported DB type in DBConfig for OdaiSqliteDb: {}", db_config.m_dbType);
    throw std::invalid_argument("Unsupported DB type in DBConfig for OdaiSqliteDb");
  }
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

OdaiResult<void> OdaiSqliteDb::initialize_db()
{
  try
  {
    // Register vec extension first
    if (!register_vec_extension())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to register sqlite-vec extension");
      return unexpected_internal_error();
    }

    if (!std::filesystem::exists(m_dbConfig.m_mediaStorePath))
    {
      if (!std::filesystem::create_directories(m_dbConfig.m_mediaStorePath))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to create media store directory: {}", m_dbConfig.m_mediaStorePath);
        return unexpected_internal_error();
      }
    }

    bool initialize_schema = false;

    // Check if DB file exists
    if (!std::filesystem::exists(m_dbConfig.m_dbPath))
    {
      initialize_schema = true;
      ODAI_LOG(ODAI_LOG_INFO, "Database file does not exist. It will be created at {}", m_dbConfig.m_dbPath);
    }

    // create db object only after registering sqlite-vec extension
    m_db = std::make_unique<SQLite::Database>(m_dbConfig.m_dbPath, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
    m_db->exec("PRAGMA foreign_keys = ON");

    ODAI_LOG(ODAI_LOG_INFO, "Opened / created database successfully at {}", m_dbConfig.m_dbPath);

    if (initialize_schema)
    {
      m_db->exec(db_schema);
      ODAI_LOG(ODAI_LOG_INFO, "initialized db with schema");
    }

    return {};
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize DB : {} Error: {}", m_dbConfig.m_dbPath, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::begin_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    m_transactionDepth++;
    if (m_transactionDepth == 1)
    {
      // Start the physical transaction
      m_transaction = std::make_unique<SQLite::Transaction>(*m_db);
    }
    return {};
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
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::commit_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
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
      return {};
    }

    ODAI_LOG(ODAI_LOG_WARN, "commit_transaction called with no active transaction");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction: {}", e.what());
    // If commit fails, we leave the transaction object (destructor will rollback if reset/destroyed)
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::rollback_transaction()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    // Regardless of depth, we roll back everything
    // Destroying the Transaction object safely rolls it back if not committed
    m_transaction.reset();
    m_transactionDepth = 0;
    return {};
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to rollback transaction: {}", e.what());
    m_transactionDepth = 0;
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::register_model_files(const ModelName& name, const ModelFiles& model_file_details,
                                                    const std::string& checksums)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (checksums.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Empty checksums passed for model: {}", name);
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }

    OdaiResult<std::string> type_str = to_model_type_db_value(model_file_details.m_modelType);
    if (!type_str)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed");
      return tl::unexpected(type_str.error());
    }

    nlohmann::json j = model_file_details;
    std::string model_file_details_json = j.dump();

    SQLite::Statement insert(*m_db, "INSERT INTO models (name, file_details, checksums, type) VALUES (:name, "
                                    "jsonb(:file_details), jsonb(:checksums), :type)");
    insert.bind(":name", name);
    insert.bind(":file_details", model_file_details_json);
    insert.bind(":checksums", checksums);
    insert.bind(":type", type_str.value());

    insert.exec();

    return {};
  }
  catch (const SQLite::Exception& e)
  {
    int ext_code = e.getExtendedErrorCode();
    if (ext_code == SQLITE_CONSTRAINT_PRIMARYKEY || ext_code == SQLITE_CONSTRAINT_UNIQUE)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model already exists: {}, DB Error: {}", name, e.what());
      return tl::unexpected(OdaiResultEnum::ALREADY_EXISTS);
    }
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to register model: {}, SQLite Error: {}", name, e.what());
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to register model: {}, Error: {}", name, e.what());
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
}

OdaiResult<ModelFiles> OdaiSqliteDb::get_model_files(const ModelName& name)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    SQLite::Statement query(*m_db, "SELECT json(file_details) as file_details FROM models WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (query.executeStep())
    {
      SQLite::Column file_details_col = query.getColumn("file_details");
      nlohmann::json file_details_json = nlohmann::json::parse(file_details_col.getString());
      return file_details_json.get<ModelFiles>();
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Model not found: {}", name);
    return tl::unexpected(OdaiResultEnum::NOT_FOUND);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get model file details: {}, Error: {}", name, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<std::string> OdaiSqliteDb::get_model_checksums(const ModelName& name)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    SQLite::Statement query(*m_db, "SELECT json(checksums) as checksums FROM models WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (query.executeStep())
    {
      return query.getColumn("checksums").getString();
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Model checksums not found: {}", name);
    return tl::unexpected(OdaiResultEnum::NOT_FOUND);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get model checksums: {}, Error: {}", name, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::update_model_files(const ModelName& name, const ModelFiles& new_model_file_details,
                                                  const std::string& new_checksums)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (new_checksums.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Empty checksums passed for model: {}", name);
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }

    OdaiResult<std::string> type_str = to_model_type_db_value(new_model_file_details.m_modelType);
    if (!type_str)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed for update");
      return tl::unexpected(type_str.error());
    }

    nlohmann::json j = new_model_file_details;
    std::string new_model_file_details_json = j.dump();

    SQLite::Statement update(*m_db, "UPDATE models SET file_details = jsonb(:file_details), "
                                    "checksums = jsonb(:checksums), type = :type WHERE name = :name");
    update.bind(":file_details", new_model_file_details_json);
    update.bind(":checksums", new_checksums);
    update.bind(":type", type_str.value());
    update.bind(":name", name);

    int rows_modified = update.exec();
    if (rows_modified == 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "No model found to update for: {}", name);
      return tl::unexpected(OdaiResultEnum::NOT_FOUND);
    }

    return {};
  }
  catch (const SQLite::Exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to update file details for model: {}, SQLite Error: {}", name, e.what());
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to update file details for model: {}, Error: {}", name, e.what());
    return tl::unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }
}

OdaiResult<InputItem> OdaiSqliteDb::store_media_item_impl(const InputItem& item, const std::string& checksum)
{
  try
  {
    // store the media item in cache dir and the mapping in db
    const std::string& file_name = checksum; // using checksum as file name to avoid duplicates
    std::string file_path = m_dbConfig.m_mediaStorePath + "/" + file_name;

    if (item.m_type == InputItemType::FILE_PATH)
    {
      // copy the file to cache dir
      std::filesystem::copy_file(byte_vector_to_string(item.m_data), file_path,
                                 std::filesystem::copy_options::overwrite_existing);
    }
    else if (item.m_type == InputItemType::MEMORY_BUFFER)
    {
      // write the buffer to a file in cache dir
      std::ofstream out_file(file_path, std::ios::binary);
      out_file.write(reinterpret_cast<const char*>(item.m_data.data()), item.m_data.size());
      out_file.close();
    }
    else
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItem type for storing media item");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    // insert the mapping in db
    SQLite::Statement insert(*m_db, "INSERT INTO media_cache (hash_xxhash, mime_type, absolute_path, file_name) VALUES "
                                    "(:checksum, :mime_type, :absolute_path, :file_name)");
    insert.bind(":checksum", checksum);
    insert.bind(":mime_type", item.m_mimeType);
    insert.bind(":absolute_path", file_path);
    insert.bind(":file_name", file_name);
    insert.exec();

    InputItem item_out;
    item_out.m_mimeType = item.m_mimeType;
    item_out.m_type = InputItemType::FILE_PATH;
    item_out.m_data = std::vector<uint8_t>(file_path.begin(), file_path.end());

    return item_out;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to store media item: {}", e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<InputItem> OdaiSqliteDb::store_media_item(const InputItem& item)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (!item.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid input item passed");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    MediaType media_type = item.get_media_type();

    if (media_type == MediaType::IMAGE || media_type == MediaType::AUDIO)
    {
      OdaiResult<std::string> checksum_res = unexpected_internal_error();

      if (item.m_type == InputItemType::FILE_PATH)
      {
        checksum_res = calculate_file_checksum(byte_vector_to_string(item.m_data));
      }
      else if (item.m_type == InputItemType::MEMORY_BUFFER)
      {
        checksum_res = calculate_data_checksum(item.m_data);
      }
      else
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItem type for storing media item");
        return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
      }

      if (!checksum_res)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate media checksum, error code: {}",
                 static_cast<std::uint32_t>(checksum_res.error()));
        return tl::unexpected(checksum_res.error());
      }
      const std::string& checksum = checksum_res.value();

      // check in db if we already have a mapping for this checksum, if yes return that path instead of storing again
      SQLite::Statement query(
          *m_db, "SELECT mime_type, absolute_path FROM media_cache WHERE hash_xxhash = :hash_xxhash LIMIT 1");
      query.bind(":hash_xxhash", checksum);

      if (query.executeStep())
      {
        std::string abs_path = query.getColumn("absolute_path").getString();
        InputItem item_out;
        item_out.m_type = InputItemType::FILE_PATH;
        item_out.m_data = std::vector<uint8_t>(abs_path.begin(), abs_path.end());
        item_out.m_mimeType = query.getColumn("mime_type").getString();
        return item_out;
      }

      // store the media item in cache dir and the mapping in db
      return store_media_item_impl(item, checksum);
    }
    if (media_type == MediaType::TEXT && item.m_type == InputItemType::MEMORY_BUFFER)
    {
      return item; // for text media type with memory buffer, we don't need to store, we can directly use the buffer as
                   // is
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Unsupported media / input type for item with mime type: {}", item.m_mimeType);
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to store media item: {}", e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::create_semantic_space(const SemanticSpaceConfig& config)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (!config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid semantic space config passed");
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }

    nlohmann::json j = config;
    std::string config_json = j.dump();

    SQLite::Statement insert(*m_db, "INSERT INTO semantic_spaces (name, config) VALUES (:name, jsonb(:config))");
    insert.bind(":name", config.m_name);
    insert.bind(":config", config_json);

    insert.exec();

    return {};
  }
  catch (const SQLite::Exception& e)
  {
    int ext_code = e.getExtendedErrorCode();
    if (ext_code == SQLITE_CONSTRAINT_PRIMARYKEY || ext_code == SQLITE_CONSTRAINT_UNIQUE)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Semantic space already exists: {}, DB Error: {}", config.m_name, e.what());
      return tl::unexpected(OdaiResultEnum::ALREADY_EXISTS);
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create semantic space: {}, SQLite Error: {}", config.m_name, e.what());
    return unexpected_internal_error();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create semantic space: {}, Error: {}", config.m_name, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<SemanticSpaceConfig> OdaiSqliteDb::get_semantic_space_config(const SemanticSpaceName& name)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    SQLite::Statement query(*m_db, "SELECT json(config) as config FROM semantic_spaces WHERE name = :name LIMIT 1");
    query.bind(":name", name);

    if (!query.executeStep())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Semantic space not found: {}", name);
      return tl::unexpected(OdaiResultEnum::NOT_FOUND);
    }

    SQLite::Column config_col = query.getColumn("config");
    nlohmann::json config_json = nlohmann::json::parse(config_col.getString());
    return config_json.get<SemanticSpaceConfig>();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get semantic space config: {}, Error: {}", name, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<std::vector<SemanticSpaceConfig>> OdaiSqliteDb::list_semantic_spaces()
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    std::vector<SemanticSpaceConfig> spaces;
    spaces.clear();

    SQLite::Statement query(*m_db, "SELECT json(config) as config FROM semantic_spaces ORDER BY name");

    while (query.executeStep())
    {
      SQLite::Column config_col = query.getColumn("config");
      nlohmann::json config_json = nlohmann::json::parse(config_col.getString());
      spaces.push_back(config_json.get<SemanticSpaceConfig>());
    }

    return spaces;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to list semantic spaces, Error: {}", e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::delete_semantic_space(const SemanticSpaceName& name)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    SQLite::Statement query(*m_db, "DELETE FROM semantic_spaces WHERE name = :name");
    query.bind(":name", name);

    if (query.exec() == 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Semantic space not found for deletion: {}", name);
      return tl::unexpected(OdaiResultEnum::NOT_FOUND);
    }

    return {};
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to delete semantic space: {}, Error: {}", name, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<bool> OdaiSqliteDb::chat_id_exists(const ChatId& chat_id)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    // "SELECT 1" is enough. We limit to 1 so the DB stops searching immediately.
    SQLite::Statement select_chat(*m_db, "SELECT 1 FROM chats WHERE chat_id = :chat_id LIMIT 1");

    // Bind the parameter, use named parameters instead of index based to avoid unexpected behaviour due to changes in
    // future
    select_chat.bind(":chat_id", chat_id);
    return select_chat.executeStep();
  }
  catch (const SQLite::Exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Check Exists SQLite Error for chat_id {}: {}", chat_id, e.what());
    return unexpected_internal_error();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Check Exists Error for chat_id {}: {}", chat_id, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::create_chat(const ChatId& chat_id, const ChatConfig& chat_config)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (!chat_config.is_sane())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid chat config passed");
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }

    nlohmann::json j = chat_config;
    std::string chat_config_json = j.dump();

    OdaiResult<void> begin_res = begin_transaction();
    if (!begin_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to begin transaction for chat creation, error code: {}",
               static_cast<std::uint32_t>(begin_res.error()));
      return tl::unexpected(begin_res.error());
    }

    try
    {
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
           std::vector<uint8_t>(chat_config.m_systemPrompt.begin(), chat_config.m_systemPrompt.end()), "text/plain"});
      system_msg.m_messageMetadata = {};

      OdaiResult<void> insert_res = insert_chat_messages(chat_id, {system_msg});
      if (!insert_res)
      {
        OdaiResult<void> rollback_res = rollback_transaction();
        if (!rollback_res)
        {
          ODAI_LOG(ODAI_LOG_WARN, "Rollback after failed chat creation insert also failed with error code: {}",
                   static_cast<std::uint32_t>(rollback_res.error()));
        }
        return insert_res;
      }

      OdaiResult<void> commit_res = commit_transaction();
      if (!commit_res)
      {
        OdaiResult<void> rollback_res = rollback_transaction();
        if (!rollback_res)
        {
          ODAI_LOG(ODAI_LOG_WARN, "Rollback after failed chat creation commit also failed with error code: {}",
                   static_cast<std::uint32_t>(rollback_res.error()));
        }
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction for chat creation, error code: {}",
                 static_cast<std::uint32_t>(commit_res.error()));
        return tl::unexpected(commit_res.error());
      }

      return {};
    }
    catch (...)
    {
      OdaiResult<void> rollback_res = rollback_transaction();
      if (!rollback_res)
      {
        ODAI_LOG(ODAI_LOG_WARN, "Rollback during chat creation exception path failed with error code: {}",
                 static_cast<std::uint32_t>(rollback_res.error()));
      }
      throw;
    }
  }
  catch (const SQLite::Exception& e)
  {
    int ext_code = e.getExtendedErrorCode();
    if (ext_code == SQLITE_CONSTRAINT_PRIMARYKEY || ext_code == SQLITE_CONSTRAINT_UNIQUE)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Chat already exists: {}, SQLite Error: {}", chat_id, e.what());
      return tl::unexpected(OdaiResultEnum::ALREADY_EXISTS);
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create chat session, SQLite Error: {}", e.what());
    return unexpected_internal_error();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create chat session Error: {}", e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<ChatConfig> OdaiSqliteDb::get_chat_config(const ChatId& chat_id)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    // always use named fields to extract data instead of index
    SQLite::Statement query(
        *m_db, "SELECT title, json(chat_config) as chat_config FROM chats WHERE chat_id = :chat_id LIMIT 1");

    query.bind(":chat_id", chat_id);

    if (!query.executeStep())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} does not exist", chat_id);
      return tl::unexpected(OdaiResultEnum::NOT_FOUND);
    }

    SQLite::Column chat_config_col = query.getColumn("chat_config");

    if (chat_config_col.isNull())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "chat_config is null for chat_id {}", chat_id);
      return unexpected_internal_error();
    }

    nlohmann::json chat_config_json = nlohmann::json::parse(chat_config_col.getString());

    return chat_config_json.get<ChatConfig>();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to Load Chat, Chat Id : {}, Error: {}", chat_id, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<std::vector<ChatMessage>> OdaiSqliteDb::get_chat_history(const ChatId& chat_id)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    std::vector<ChatMessage> messages;
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

      nlohmann::json content_json = nlohmann::json::parse(content_col.getString());
      msg.m_contentItems = content_json.get<std::vector<InputItem>>();

      // Handle NULL message_metadata by defaulting to empty JSON object
      if (metadata_col.isNull())
      {
        msg.m_messageMetadata = {};
      }
      else
      {
        msg.m_messageMetadata = nlohmann::json::parse(metadata_col.getString());
      }

      // Cast to uint64_t since Unix timestamps are always non-negative
      msg.m_createdAt = static_cast<uint64_t>(created_at_col.getInt64());

      messages.push_back(msg);
    }

    if (!has_results)
    {
      OdaiResult<bool> exists_res = chat_id_exists(chat_id);
      if (!exists_res)
      {
        return tl::unexpected(exists_res.error());
      }

      if (!exists_res.value())
      {
        ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} does not exist", chat_id);
        return tl::unexpected(OdaiResultEnum::NOT_FOUND);
      }
    }

    return messages;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get chat history, Chat Id : {}, Error: {}", chat_id, e.what());
    return unexpected_internal_error();
  }
}

OdaiResult<void> OdaiSqliteDb::insert_chat_messages(const ChatId& chat_id, const std::vector<ChatMessage>& messages)
{
  try
  {
    if (m_db == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
      return unexpected_not_initialized();
    }

    if (messages.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "no messages passed to insert for chat_id: {}", chat_id);
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }

    // Start transaction (supports nesting)
    OdaiResult<void> begin_res = begin_transaction();
    if (!begin_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to begin transaction for inserting chat messages, error code: {}",
               static_cast<std::uint32_t>(begin_res.error()));
      return tl::unexpected(begin_res.error());
    }

    try
    {
      // Prepare statement once, reuse for all messages
      SQLite::Statement insert_message(
          *m_db, "INSERT INTO chat_messages (chat_id, role, content, message_metadata, sequence_index) "
                 "VALUES (:chat_id, :role, :content, jsonb(:message_metadata), "
                 "COALESCE((SELECT MAX(sequence_index) + 1 FROM chat_messages WHERE chat_id = :chat_id), 0))");

      // this fn is given messages where message content items are coming from db's store_media_items

      for (const auto& msg : messages)
      {

        for (const auto& item : msg.m_contentItems)
        {
          MediaType media_type = item.get_media_type();

          if (media_type == MediaType::TEXT && item.m_type != InputItemType::MEMORY_BUFFER)
          {
            ODAI_LOG(ODAI_LOG_ERROR, "Text should only be passed as memory buffer");
            OdaiResult<void> rollback_res = rollback_transaction();
            if (!rollback_res)
            {
              ODAI_LOG(ODAI_LOG_WARN, "Rollback after invalid text item failed with error code: {}",
                       static_cast<std::uint32_t>(rollback_res.error()));
            }
            return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
          }

          if ((media_type == MediaType::AUDIO || media_type == MediaType::IMAGE) &&
              (item.m_type != InputItemType::FILE_PATH))
          {
            ODAI_LOG(ODAI_LOG_ERROR, "Media items should only be passed as file paths");
            OdaiResult<void> rollback_res = rollback_transaction();
            if (!rollback_res)
            {
              ODAI_LOG(ODAI_LOG_WARN, "Rollback after invalid media item failed with error code: {}",
                       static_cast<std::uint32_t>(rollback_res.error()));
            }
            return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
          }
        }

        insert_message.bind(":chat_id", chat_id);
        insert_message.bind(":role", msg.m_role);
        nlohmann::json content_json = msg.m_contentItems;
        insert_message.bind(":content", content_json.dump());
        insert_message.bind(":message_metadata", msg.m_messageMetadata.dump());
        insert_message.exec();
        insert_message.reset(); // Reset for next iteration
        insert_message.clearBindings();
      }

      OdaiResult<void> commit_res = commit_transaction();
      if (!commit_res)
      {
        OdaiResult<void> rollback_res = rollback_transaction();
        if (!rollback_res)
        {
          ODAI_LOG(ODAI_LOG_WARN, "Rollback after failed message insert commit also failed with error code: {}",
                   static_cast<std::uint32_t>(rollback_res.error()));
        }
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction for inserting chat messages, error code: {}",
                 static_cast<std::uint32_t>(commit_res.error()));
        return tl::unexpected(commit_res.error());
      }
    }
    catch (...)
    {
      OdaiResult<void> rollback_res = rollback_transaction();
      if (!rollback_res)
      {
        ODAI_LOG(ODAI_LOG_WARN, "Rollback during insert_chat_messages exception path failed with error code: {}",
                 static_cast<std::uint32_t>(rollback_res.error()));
      }
      throw; // Re-throw to be caught by outer catch
    }

    return {};
  }
  catch (const SQLite::Exception& e)
  {
    int ext_code = e.getExtendedErrorCode();
    if (ext_code == SQLITE_CONSTRAINT_FOREIGNKEY)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Chat not found while inserting messages for chat_id {}: {}", chat_id, e.what());
      return tl::unexpected(OdaiResultEnum::NOT_FOUND);
    }

    ODAI_LOG(ODAI_LOG_ERROR, "Failed to insert chat messages SQLite Error: {}", e.what());
    return unexpected_internal_error();
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to insert chat messages Error: {}", e.what());
    return unexpected_internal_error();
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
