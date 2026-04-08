# OdaiSqliteDb — SQLite Database Implementation

**Interface**: [`IOdaiDb`](../interfaces/database.md)  
**Header**: [`src/include/db/odai_sqlite/odai_sqlite_db.h`](../../src/include/db/odai_sqlite/odai_sqlite_db.h)  
**Implementation**: `src/impl/db/odai_sqlite/`  
**CMake Guard**: `ODAI_ENABLE_SQLITE_DB`

## Overview

Uses SQLite (via [SQLiteCpp](https://github.com/SRombauts/SQLiteCpp) wrapper) with the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension for vector similarity search. Single-file database suitable for on-device deployment.

## Build Integration

- **SQLiteCpp** — Fetched via CMake `FetchContent`, built as a library, linked to `odai`.
- **sqlite-vec** — Amalgamation included in repo at `sqlite-vec-0.1.6-amalgamation/`. Automating its fetch is a TODO item.
- The `sqlite-vec` extension is registered before opening the database.

## Schema Overview

The schema (defined inline in the header as `db_schema`) includes:

| Table | Purpose |
|---|---|
| `chats` | Chat session metadata + config (JSON blob) |
| `chat_messages` | Messages with role, content, sequence order, metadata |
| `media_cache` | Maps XXHash checksums to cached file paths |
| `document` | Source documents for RAG (with scope partitioning) |
| `chunk` | Deduplicated content chunks (hash-based dedup) |
| `doc_chunk_ref` | Many-to-many link between documents and chunks |
| `semantic_spaces` | Semantic space configs (JSON blob) |
| `models` | Registered model names, file details, checksums, type |

The `vec_items` virtual table (sqlite-vec) for vector search is defined but commented out pending RAG pipeline completion.

## Transaction Handling

Implements flattened nested transactions using a depth counter — real SQL transaction starts on the first `begin_transaction()`, real commit happens only on the outermost `commit_transaction()`, and `rollback_transaction()` always does a full abort regardless of depth.

## Media Caching

Media items are deduplicated by XXHash checksum and cached to `m_mediaStorePath`. If a checksum already exists, the existing path is returned without re-storing. Text items skip storage entirely.

## Known Limitations

- **Not thread-safe** — `Database`, `Statement`, and `Transaction` objects cannot be shared across threads. Would need one DB object per thread or mutex locks.
- Vector store table (`vec_items`) is not yet active
