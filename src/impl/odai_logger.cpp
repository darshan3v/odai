#include "odai_logger.h"

using namespace std;

void ODAILogger::odai_set_logger(odai_log_callback_fn callback, void *user_data)
{
    this->callback = callback;
    this->user_data = user_data;
}

void ODAILogger::odai_set_log_level(OdaiLogLevel log_level)
{
    this->log_level = log_level;
}