#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "utils/odai_helpers.h"
#include "xxhash.h"

using namespace std;

ChatId generate_chat_id()
{
  // Simple random ID generation
  return string("chat_") + to_string(rand()) + "_" + "t" + to_string(time(nullptr));
}

string calculate_file_checksum(const string& path)
{
  ifstream file(path, ios::binary);
  if (!file.is_open())
  {
    return "";
  }

  // Use XXH3_64bits for speed and simple checksum
  XXH3_state_t* state = XXH3_createState();
  if (state == nullptr)
  {
    return "";
  }

  XXH3_64bits_reset(state);

  const size_t BUFFER_SIZE = 64 * 1024; // 64KB buffer
  vector<char> buffer(BUFFER_SIZE);

  while (file.read(buffer.data(), BUFFER_SIZE))
  {
    XXH3_64bits_update(state, buffer.data(), file.gcount());
  }
  // Handle remaining bytes
  XXH3_64bits_update(state, buffer.data(), file.gcount());

  XXH64_hash_t hash = XXH3_64bits_digest(state);
  XXH3_freeState(state);

  stringstream ss;
  ss << hex << setw(16) << setfill('0') << hash;
  return ss.str();
}