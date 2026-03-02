#pragma once

#include "types/odai_types.h"

using namespace std;

/// Generates a unique chat session identifier.
/// The ID is composed of a "chat_" prefix, a random number, and the current timestamp.
/// @return A unique string identifier in the format "chat_<random>_<timestamp>"
ChatId generate_chat_id();

string calculate_file_checksum(const string& path);

/// Calculates checksums for all files in a ModelFiles struct.
/// @param files The Model Files struct containing paths.
/// @return A JSON string containing key-value pairs of checksums, or empty string on error.
string calculate_model_checksums(const ModelFiles& files);