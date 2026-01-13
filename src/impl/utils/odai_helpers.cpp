#include <ctime>

#include "utils/odai_helpers.h"

ChatId generate_chat_id()
{
    return string("chat_") + to_string(rand()) + "_" + "t" + to_string(time(nullptr));
}