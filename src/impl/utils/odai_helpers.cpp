#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "xxhash.h"
#include "utils/odai_helpers.h"

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

    const size_t buffer_size = 64 * 1024; // 64KB buffer
    vector<char> buffer(buffer_size);

    while (file.read(buffer.data(), buffer_size))
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