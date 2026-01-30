#include "odai_sdk.h"
#include "db/odai_sqlite_db.h"

#include <sqlite3.h>
#include <nlohmann/json.hpp>
#include <filesystem>

#include "types/odai_type_conversions.h"

using namespace nlohmann;

// SQLiteCpp uses try catch handling heavily so we use them here a lot

using namespace std;

extern "C" int sqlite3_vec_init(sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

ODAISqliteDb::ODAISqliteDb(const DBConfig &dbConfig)
{
    this->dbPath = dbConfig.dbPath;
}

bool ODAISqliteDb::register_vec_extension()
{
    try
    {
        // because of the way sqlite3 is built, can't directly call sqlite3_vec_init, if you call it will crash
        int ret_code = sqlite3_auto_extension((void (*)(void))sqlite3_vec_init);

        if (ret_code != SQLITE_OK)
        {
            const char *error_msg = sqlite3_errstr(ret_code);
            ODAI_LOG(ODAI_LOG_ERROR, "Failed to register sqlite vec extension, code: {}, error : {}", ret_code, error_msg);
            return false;
        }

        ODAI_LOG(ODAI_LOG_INFO, "sqlite-vec extension registered successfully");

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "DB Init Error: {}", e.what());
        return false;
    }
}

bool ODAISqliteDb::initialize_db()
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
        if (!filesystem::exists(dbPath))
        {
            initialize_schema = true;
            ODAI_LOG(ODAI_LOG_INFO, "Database file does not exist. It will be created at {}", dbPath);
        }

        // create db object only after registering sqlite-vec extension
        db = std::make_unique<SQLite::Database>(
            dbPath,
            SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);

        ODAI_LOG(ODAI_LOG_INFO, "Opened / created database successfully at {}", dbPath);

        if (initialize_schema)
        {
        db->exec(DB_SCHEMA);
        ODAI_LOG(ODAI_LOG_INFO, "initialized db with schema");
        }

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize DB : {} Error: {}", dbPath, e.what());
        return false;
    }
}

bool ODAISqliteDb::chat_id_exists(const ChatId &chat_id)
{
    try
    {
        // "SELECT 1" is enough. We limit to 1 so the DB stops searching immediately.
        SQLite::Statement select_chat(*db, "SELECT 1 FROM chats WHERE chat_id = :chat_id LIMIT 1");

        // Bind the parameter, use named parameters instead of index based to avoid unexpected behaviour due to changes in future
        select_chat.bind(":chat_id", chat_id);
        select_chat.exec();

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Check Exists Error: {}", e.what());
        return false;
    }
}

bool ODAISqliteDb::create_chat(const ChatId &chat_id, const ChatConfig &chat_config)
{
    try
    {

        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        if(!chat_config.is_sane())
        {
            ODAI_LOG(ODAI_LOG_ERROR,"Invalid chat config passed");
            return false;
        }

        json j = chat_config;
        string chat_config_json = j.dump();

        begin_transaction();

        SQLite::Statement insert_chat(*db, "INSERT INTO chats (chat_id, chat_config) VALUES (:chat_id, jsonb(:chat_config))");

        insert_chat.bind(":chat_id", chat_id);
        insert_chat.bind(":chat_config", chat_config_json);
        insert_chat.exec();

        // Insert system prompt as initial message
        ChatMessage system_msg;
        system_msg.role = "system";
        system_msg.content = chat_config.system_prompt;
        system_msg.message_metadata = {};
        insert_chat_messages(chat_id, {system_msg});

        commit_transaction();

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to create chat session Error: {}", e.what());
        return false;
    }
}

bool ODAISqliteDb::get_chat_config(const ChatId &chat_id, ChatConfig &chat_config)
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        // always use named fields to extract data instead of index
        SQLite::Statement query(*db, "SELECT title, json(chat_config) as chat_config FROM chats WHERE chat_id = :chat_id LIMIT 1");

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
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to Load Chat, Chat Id : {}, Error: {}", chat_id, e.what());
        return false;
    }
}

bool ODAISqliteDb::get_chat_history(const ChatId &chat_id, vector<ChatMessage> &messages)
{
    try
    {

        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        messages.clear();

        SQLite::Statement query(*db,
                                "SELECT role, content, json(message_metadata) as message_metadata, created_at "
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

            msg.role = role_col.getString();
            msg.content = content_col.getString();

            // Handle NULL message_metadata by defaulting to empty JSON object
            if (metadata_col.isNull())
            {
                msg.message_metadata = {};
            }
            else
            {
                msg.message_metadata = json::parse(metadata_col.getString());
            }

            // Cast to uint64_t since Unix timestamps are always non-negative
            msg.created_at = static_cast<uint64_t>(created_at_col.getInt64());

            messages.push_back(msg);
        }

        if (!has_results)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "chat_id {} does not exist or has no messages", chat_id);
            return false;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to get chat history, Chat Id : {}, Error: {}", chat_id, e.what());
        return false;
    }
}

ODAISqliteDb::~ODAISqliteDb()
{
}

bool ODAISqliteDb::insert_chat_messages(const ChatId &chat_id, const vector<ChatMessage> &messages)
{
    try
    {
        if (db == nullptr)
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

        try {
            // Prepare statement once, reuse for all messages
            SQLite::Statement insert_message(*db,
                                             "INSERT INTO chat_messages (chat_id, role, content, message_metadata, sequence_index) "
                                             "VALUES (:chat_id, :role, :content, jsonb(:message_metadata), "
                                             "COALESCE((SELECT MAX(sequence_index) + 1 FROM chat_messages WHERE chat_id = :chat_id), 0))");

            for (const auto &msg : messages)
            {
                insert_message.bind(":chat_id", chat_id);
                insert_message.bind(":role", msg.role);
                insert_message.bind(":content", msg.content);
                insert_message.bind(":message_metadata", msg.message_metadata.dump());
                insert_message.exec();
                insert_message.reset(); // Reset for next iteration
                insert_message.clearBindings();
            }
            
            commit_transaction();
        }
        catch (...) {
            rollback_transaction();
            throw; // Re-throw to be caught by outer catch
        }

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to insert chat messages Error: {}", e.what());
        return false;
    }
}

void ODAISqliteDb::close()
{
    try
    {
        if (db != nullptr)
        {
            db.reset();
            ODAI_LOG(ODAI_LOG_INFO, "Database connection closed successfully");
        }
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Error closing database: {}", e.what());
    }
}

bool ODAISqliteDb::begin_transaction()
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        m_transaction_depth++;
        if (m_transaction_depth == 1)
        {
            // Start the physical transaction
            m_transaction = std::make_unique<SQLite::Transaction>(*db);
        }
        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to begin transaction: {}", e.what());
        if (m_transaction_depth == 1) 
        {
             m_transaction_depth = 0;
             m_transaction.reset();
        }
        else m_transaction_depth--; 
        return false;
    }
}

bool ODAISqliteDb::commit_transaction()
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        if (m_transaction_depth > 0)
        {
            m_transaction_depth--;
            if (m_transaction_depth == 0)
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
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to commit transaction: {}", e.what());
        // If commit fails, we leave the transaction object (destructor will rollback if reset/destroyed)
        return false;
    }
}

bool ODAISqliteDb::rollback_transaction()
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        // Regardless of depth, we roll back everything
        // Destroying the Transaction object safely rolls it back if not committed
        m_transaction.reset();
        m_transaction_depth = 0;
        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to rollback transaction: {}", e.what());
        m_transaction_depth = 0;
        return false;
    }
}

bool ODAISqliteDb::create_semantic_space(const SemanticSpaceConfig &config)
{
    try
    {
        if (db == nullptr)
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

        SQLite::Statement insert(*db, "INSERT INTO semantic_spaces (name, config) VALUES (:name, jsonb(:config))");
        insert.bind(":name", config.name);
        insert.bind(":config", config_json);
        
        insert.exec();

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to create semantic space: {}, Error: {}", config.name, e.what());
        return false;
    }
}

bool ODAISqliteDb::list_semantic_spaces(vector<SemanticSpaceConfig> &spaces)
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        spaces.clear();

        SQLite::Statement query(*db, "SELECT json(config) as config FROM semantic_spaces ORDER BY name");

        while (query.executeStep())
        {
            SQLite::Column config_col = query.getColumn("config");
            json config_json = json::parse(config_col.getString());
            spaces.push_back(config_json.get<SemanticSpaceConfig>());
        }

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to list semantic spaces, Error: {}", e.what());
        return false;
    }
}

bool ODAISqliteDb::delete_semantic_space(const string &name)
{
    try
    {
        if (db == nullptr)
        {
            ODAI_LOG(ODAI_LOG_ERROR, "Database not initialized");
            return false;
        }

        SQLite::Statement remove(*db, "DELETE FROM semantic_spaces WHERE name = :name");
        remove.bind(":name", name);
        remove.exec();

        return true;
    }
    catch (const std::exception &e)
    {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to delete semantic space: {}, Error: {}", name, e.what());
        return false;
    }
}