#pragma once

#include <cstddef>
#include <string>

using namespace std;

/// Returns the length of the string that is safe to send as valid UTF-8.
/// Scans backwards from the end of the string to ensure the last character is complete.
/// If the string ends with an incomplete multi-byte UTF-8 sequence, returns the length excluding the incomplete bytes.
/// @param buffer The string buffer to check
/// @return The safe length in bytes that represents complete UTF-8 characters, or 0 if buffer is empty
size_t get_safe_utf8_length(const string& buffer);

/// Copies a C++ string to a C-style string buffer and sets the length.
/// The destination buffer must be large enough to hold the string plus null terminator.
/// @param str The source C++ string to copy
/// @param c_str Destination C-style string buffer (must be pre-allocated)
/// @param cstr_len Pointer to size_t that will be set to the required buffer size (including null terminator)
void set_cstr_and_len(string& str, char* c_str, size_t* cstr_len);