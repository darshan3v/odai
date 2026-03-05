#pragma once

#include "types/odai_types.h"

using namespace std;

/// Generates a unique chat session identifier.
/// The ID is composed of a "chat_" prefix, a random number, and the current timestamp.
/// @return A unique string identifier in the format "chat_<random>_<timestamp>"
ChatId generate_chat_id();

/// Calculates the XXH3 64-bit checksum of a file's contents.
/// @param path The absolute path to the file.
/// @return A 16-character hexadecimal string representing the checksum, or an empty string on error.
string calculate_file_checksum(const string& path);

/// Calculates the XXH3 64-bit checksum of a byte array in memory.
/// @param data The vector containing the binary data.
/// @return A 16-character hexadecimal string representing the checksum, or an empty string if data is empty.
string calculate_data_checksum(const vector<uint8_t>& data);

/// Identifies the MediaType from a given mime_type string.
/// @param mime_type The mime type string to evaluate.
/// @return The appropriate MediaType enum.
MediaType get_media_type_from_mime(const std::string& mime_type);

/// Calculates checksums for all files in a ModelFiles struct.
/// @param files The Model Files struct containing paths.
/// @return A JSON string containing key-value pairs of checksums, or empty string on error.
string calculate_model_checksums(const ModelFiles& files);