#include "backendEngine/odai_llamacpp/odai_llama_type_conversions.h"

BackendDeviceType to_odai_backend_device_type(enum ggml_backend_dev_type type)
{
  switch (type)
  {
  case GGML_BACKEND_DEVICE_TYPE_CPU:
    return BackendDeviceType::CPU;
  case GGML_BACKEND_DEVICE_TYPE_GPU:
    return BackendDeviceType::GPU;
  case GGML_BACKEND_DEVICE_TYPE_IGPU:
    return BackendDeviceType::IGPU;
  default:
    return BackendDeviceType::AUTO;
  }
}
