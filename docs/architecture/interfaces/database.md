# IOdaiDb — Database Interface

**Header**: [`src/include/db/odai_db.h`](../../src/include/db/odai_db.h)

## Purpose

Abstracts all persistent storage: model registration, chat sessions, semantic spaces, media caching, and (future) vector storage. Any database backend (SQLite, PostgreSQL, etc.) implements this interface.

## Ownership

`OdaiRagEngine` owns a `std::unique_ptr<IOdaiDb>`. Created during SDK initialization.

## Responsibilities

- **Model management** — register, retrieve, and update model file records with checksums.
- **Chat sessions** — create chats, store/retrieve messages in chronological order, persist configs.
- **Semantic spaces** — CRUD for named knowledge domains with embedding model + chunking strategy configs.
- **Media caching** — store media items (images/audio) to disk, deduplicate by checksum, return file paths.
- **Transactions** — support flattened nested transactions (depth counter pattern).

## Important Behavioral Contracts

- **`update_model_files` is a full replace** — the caller (RAG engine) merges old + new details before calling. The DB layer overwrites the entire record.
- **Media items flow** — before `insert_chat_messages`, callers must `store_media_item()` for each media item to get its cached file path. Text items (`MEMORY_BUFFER`) skip storage.
- **Chat history retrieval** — media items are returned as `FILE_PATH` pointing to cached files, not raw binary data.

## Current Implementation

- [OdaiSqliteDb (SQLite + sqlite-vec)](../implementations/sqlite-database.md)
