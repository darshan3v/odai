#include "odai_logger.h"

using namespace std;

void ODAILogger::set_logger(OdaiLogCallbackFn callback, void* user_data)
{
  this->m_callback = callback;
  this->m_userData = user_data;
}

void ODAILogger::set_log_level(OdaiLogLevel log_level)
{
  this->m_logLevel = log_level;
}