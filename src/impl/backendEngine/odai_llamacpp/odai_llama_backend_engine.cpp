#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include "audioEngine/odai_audio_decoder.h"
#include "backendEngine/odai_llamacpp/odai_llama_backend_engine.h"
#include "backendEngine/odai_llamacpp/odai_llama_type_conversions.h"
#include "imageEngine/odai_image_decoder.h"

#include "odai_sdk.h"

#include "types/odai_common_types.h"
#include "types/odai_type_conversions.h"
#include "types/odai_types.h"
#include "utils/odai_exception_macros.h"
#include "utils/odai_helpers.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
std::string format_backend_list(const std::vector<std::string_view>& backends)
{
  const std::string joined = join_strings(backends, ", ");
  return joined.empty() ? "(none)" : joined;
}

std::vector<std::string_view> get_prioritized_gpu_backends()
{
#ifdef __APPLE__
  return {"metal"};
#elif defined(__ANDROID__)
  return {"vulkan"};
#else
  return {"cuda", "hip", "sycl", "vulkan"};
#endif
}

std::vector<std::string_view> get_prioritized_igpu_backends()
{
#ifdef __APPLE__
  return {};
#elif defined(__ANDROID__)
  return {"vulkan"};
#else
  return {"sycl", "vulkan"};
#endif
}

const char* device_type_name(BackendDeviceType type)
{
  switch (type)
  {
  case BackendDeviceType::AUTO:
    return "AUTO";
  case BackendDeviceType::CPU:
    return "CPU";
  case BackendDeviceType::GPU:
    return "GPU";
  case BackendDeviceType::IGPU:
    return "IGPU";
  default:
    return "UNKNOWN";
  }
}

bool contains_type(const std::vector<BackendDeviceType>& accepted_types, BackendDeviceType type)
{
  return std::find(accepted_types.begin(), accepted_types.end(), type) != accepted_types.end();
}

bool is_accelerated_device_type(BackendDeviceType type)
{
  return (type == BackendDeviceType::GPU) || (type == BackendDeviceType::IGPU);
}

constexpr int32_t N_GPU_LAYERS_ALL_POSSIBLE = -1;
constexpr uint64_t LLM_VRAM_OVERHEAD_BUFFER = 350ULL * BYTES_PER_MB;
constexpr uint64_t CPU_MLOCK_HEADROOM = 2ULL * BYTES_PER_GB;
} // namespace

/// Redirects llama.cpp log messages to the Odai logging system.
/// Maps GGML log levels to Odai log levels and filters out debug messages.
/// @param level The GGML log level
/// @param text The log message text
/// @param user_data User data (unused)
static void llama_log_redirect(ggml_log_level level, const char* text, void* user_data)
{
  // Map llama levels to Odai levels
  OdaiLogLevel our_level = ODAI_LOG_INFO;

  if (level == GGML_LOG_LEVEL_ERROR)
  {
    our_level = ODAI_LOG_ERROR;
  }
  else if (level == GGML_LOG_LEVEL_WARN)
  {
    our_level = ODAI_LOG_WARN;
  }
  else if (level == GGML_LOG_LEVEL_INFO)
  {
    our_level = ODAI_LOG_INFO;
  }
  else
  {
    return; // Ignore debug/spam from llama if you want
  }

  ODAI_LOG(our_level, "[llama.cpp] {}", text);
}

OdaiLlamaEngine::OdaiLlamaEngine(const BackendEngineConfig& backend_engine_config)
    : IOdaiBackendEngine(backend_engine_config)
{
  if (backend_engine_config.m_engineType != LLAMA_BACKEND_ENGINE)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid BackendEngineConfig provided to OdaiLlamaEngine constructor");
    throw std::invalid_argument("Invalid BackendEngineConfig provided to OdaiLlamaEngine constructor");
  }
}

bool OdaiLlamaEngine::register_available_backends()
{
  const std::filesystem::path backend_dir =
      get_module_directory_from_address(reinterpret_cast<const void*>(&register_available_backends));
  const std::string backend_dir_str = backend_dir.string();

  ODAI_LOG(ODAI_LOG_INFO, "Registering ggml backends from runtime directory: {}", backend_dir_str);
  ggml_backend_load_all_from_path(backend_dir_str.c_str());

  if (ggml_backend_reg_by_name("cpu") != nullptr)
  {
    return true;
  }

  ODAI_LOG(ODAI_LOG_ERROR, "Failed to register the CPU backend from runtime directory: {}", backend_dir_str);
  return false;
}

OdaiResult<void> OdaiLlamaEngine::discover_candidate_devices(BackendDeviceType preferred_type)
{
  try
  {
    m_deviceInventory = {};
    m_deviceInventory.m_requestedType = preferred_type;

    if (!OdaiLlamaEngine::register_available_backends())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Critical failure: CPU backend library not found");
      return unexpected_internal_error<void>();
    }

    ODAI_LOG(ODAI_LOG_INFO, "Discovering candidate devices for preferred type {}", device_type_name(preferred_type));

    auto append_candidates_with_log = [this](const std::string_view backend_name,
                                             const std::vector<BackendDeviceType>& accepted_types) -> bool
    {
      const std::string accepted_type_names =
          join_strings(accepted_types, ", ", [](BackendDeviceType type) { return device_type_name(type); });

      ODAI_LOG(ODAI_LOG_INFO, "Probing backend family '{}' for device types [{}]", backend_name, accepted_type_names);
      return this->append_backend_candidate_devices(std::string(backend_name), accepted_types);
    };

    if (preferred_type == BackendDeviceType::CPU)
    {
      ODAI_LOG(ODAI_LOG_INFO, "CPU only mode requested; discovery will probe the CPU backend directly");
      if (append_candidates_with_log("cpu", {BackendDeviceType::CPU}))
      {
        ODAI_LOG(ODAI_LOG_INFO, "Discovered {} CPU candidate device(s)", m_deviceInventory.m_candidates.size());
        return {};
      }

      ODAI_LOG(ODAI_LOG_ERROR, "CPU backend was registered but no CPU device was discovered");
      return tl::make_unexpected(OdaiResultEnum::NOT_FOUND);
    }

    const std::vector<std::string_view> gpu_backends = get_prioritized_gpu_backends();
    const std::vector<std::string_view> igpu_backends = get_prioritized_igpu_backends();
    bool candidates_found = false;

    if (preferred_type == BackendDeviceType::GPU || preferred_type == BackendDeviceType::AUTO)
    {
      ODAI_LOG(ODAI_LOG_INFO, "Prioritized GPU backend probe order: {}", format_backend_list(gpu_backends));

      for (const std::string_view backend_name : gpu_backends)
      {
        std::vector<BackendDeviceType> accepted_types = {BackendDeviceType::GPU};

#ifdef __ANDROID__
        if (backend_name == "vulkan")
        {
          accepted_types.push_back(BackendDeviceType::IGPU);
        }
#endif

        if (append_candidates_with_log(backend_name, accepted_types))
        {
          m_deviceInventory.m_hasAccelerationCandidate = true;
          candidates_found = true;

#ifdef __ANDROID__
          if (contains_type(accepted_types, BackendDeviceType::IGPU))
          {
            ODAI_LOG(ODAI_LOG_INFO,
                     "Accepted Android Vulkan integrated devices as acceleration candidates during GPU probing");
          }
#endif

          break;
        }
      }
    }

    if (preferred_type == BackendDeviceType::IGPU)
    {
      ODAI_LOG(ODAI_LOG_INFO, "Prioritized IGPU backend probe order: {}", format_backend_list(igpu_backends));

      if (igpu_backends.empty())
      {
        ODAI_LOG(ODAI_LOG_INFO, "No dedicated IGPU backend probe is configured for this platform");
      }

      for (const std::string_view backend_name : igpu_backends)
      {
        if (append_candidates_with_log(backend_name, {BackendDeviceType::IGPU}))
        {
          m_deviceInventory.m_hasAccelerationCandidate = true;
          candidates_found = true;
          break;
        }
      }
    }

    if (!candidates_found && preferred_type == BackendDeviceType::AUTO)
    {
      ODAI_LOG(ODAI_LOG_INFO, "AUTO mode did not find an accelerated GPU candidate; discovery will fall back to CPU");

      if (append_candidates_with_log("cpu", {BackendDeviceType::CPU}))
      {
        candidates_found = true;
        ODAI_LOG(ODAI_LOG_INFO, "Discovered {} CPU fallback candidate device(s)",
                 m_deviceInventory.m_candidates.size());
      }
      else
      {
        ODAI_LOG(ODAI_LOG_ERROR, "CPU backend was registered but no CPU device was discovered");
        return unexpected_internal_error<void>();
      }
    }

    if (!candidates_found && (preferred_type == BackendDeviceType::GPU || preferred_type == BackendDeviceType::IGPU))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Requested {} device not found in discovered hardware",
               preferred_type == BackendDeviceType::GPU ? "GPU" : "IGPU");
      return tl::make_unexpected(OdaiResultEnum::NOT_FOUND);
    }

    if (m_deviceInventory.m_candidates.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "No candidate devices were discovered for preferred type {}",
               device_type_name(preferred_type));
      return tl::make_unexpected(OdaiResultEnum::NOT_FOUND);
    }

    ODAI_LOG(ODAI_LOG_INFO, "Discovered {} candidate device(s) for preferred type {}",
             m_deviceInventory.m_candidates.size(), device_type_name(preferred_type));

    if (m_deviceInventory.m_hasAccelerationCandidate)
    {
      ODAI_LOG(ODAI_LOG_INFO, "Hardware acceleration candidates are available for later placement planning");
    }

    return {};
  }
  ODAI_CATCH_RETURN(unexpected_internal_error<void>())
}

bool OdaiLlamaEngine::append_backend_candidate_devices(const std::string& backend_name,
                                                       const std::vector<BackendDeviceType>& accepted_types)
{
  if (ggml_backend_reg_by_name(backend_name.c_str()) == nullptr)
  {
    ODAI_LOG(ODAI_LOG_INFO, "Backend family '{}' is not registered in this runtime", backend_name);
    return false;
  }

  bool found = false;
  size_t backend_device_count = 0;
  size_t matching_type_count = 0;
  size_t count = ggml_backend_dev_count();
  for (size_t i = 0; i < count; ++i)
  {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (to_lower(ggml_backend_reg_name(reg)) == to_lower(backend_name))
    {
      ++backend_device_count;
      BackendDeviceType dev_type = to_odai_backend_device_type(ggml_backend_dev_type(dev));
      if (contains_type(accepted_types, dev_type))
      {
        ++matching_type_count;
        CandidateDeviceRecord record;
        record.m_info.m_name = ggml_backend_dev_name(dev);
        record.m_info.m_description = ggml_backend_dev_description(dev);
        record.m_handle = dev;

        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        record.m_info.m_totalRam = total_mem;
        record.m_info.m_type = dev_type;

        m_deviceInventory.m_candidates.push_back(record);

        ODAI_LOG(
            ODAI_LOG_INFO, "Discovered candidate: {} ({}) | Backend: {} | Type: {} | Memory: {} MB free / {} MB total",
            record.m_info.m_name, record.m_info.m_description, backend_name, static_cast<uint8_t>(record.m_info.m_type),
            bytes_to_mb(free_mem), bytes_to_mb(record.m_info.m_totalRam));
        found = true;
      }
    }
  }

  ODAI_LOG(ODAI_LOG_INFO, "Probe result for backend '{}': {} device(s) in family, {} accepted candidate(s)",
           backend_name, backend_device_count, matching_type_count);
  return found;
}

std::optional<uint64_t> OdaiLlamaEngine::get_device_free_memory_bytes(ggml_backend_dev_t backend_handle)
{
  if (backend_handle == nullptr)
  {
    return std::nullopt;
  }

  size_t free_mem = 0;
  size_t total_mem = 0;
  ggml_backend_dev_memory(backend_handle, &free_mem, &total_mem);
  return static_cast<uint64_t>(free_mem);
}

OdaiLlamaEngine::LlmLoadPlan OdaiLlamaEngine::make_cpu_only_llm_plan(uint64_t model_file_size_bytes,
                                                                     std::string reason) const
{
  LlmLoadPlan plan;
  plan.m_mode = PlacementMode::CPU_ONLY;
  plan.m_nGpuLayers = 0;
  plan.m_splitMode = LLAMA_SPLIT_MODE_NONE;
  plan.m_allowCpuRetry = false;

#if defined(__APPLE__) || defined(__ANDROID__)
  plan.m_shouldUseMlock = false;
#else
  std::optional<uint64_t> system_free_ram = std::nullopt;
  for (const CandidateDeviceRecord& record : m_deviceInventory.m_candidates)
  {
    if (record.m_info.m_type != BackendDeviceType::CPU)
    {
      continue;
    }

    system_free_ram = get_device_free_memory_bytes(record.m_handle);
    break;
  }

  plan.m_shouldUseMlock =
      system_free_ram.has_value() && (system_free_ram.value() > (model_file_size_bytes + CPU_MLOCK_HEADROOM));

  if (system_free_ram.has_value())
  {
    reason += std::format(" CPU free RAM {} MB, model {} MB, mlock {}.", system_free_ram.value() / BYTES_PER_MB,
                          model_file_size_bytes / BYTES_PER_MB, plan.m_shouldUseMlock ? "enabled" : "disabled");
  }
  else
  {
    reason += " CPU free RAM unavailable, mlock disabled.";
  }
#endif

  plan.m_reason = std::move(reason);
  return plan;
}

bool OdaiLlamaEngine::build_llm_model_params(const LlmLoadPlan& llm_load_plan, llama_model_params& out_model_params,
                                             std::vector<ggml_backend_dev_t>& out_selected_devices) const
{
  out_model_params = llama_model_default_params();
  out_model_params.n_gpu_layers = llm_load_plan.m_nGpuLayers;
  out_model_params.devices = nullptr;
  out_model_params.main_gpu = llm_load_plan.m_mode == PlacementMode::CPU_ONLY ? -1 : 0;
  out_model_params.split_mode = llm_load_plan.m_splitMode;
  out_model_params.use_mlock = llm_load_plan.m_shouldUseMlock;
  out_model_params.use_mmap = llm_load_plan.m_shouldUseMmap;

  if (llm_load_plan.m_selectedCandidateIndices.empty())
  {
    return true;
  }

  out_selected_devices.clear();
  out_selected_devices.reserve(llm_load_plan.m_selectedCandidateIndices.size() + 1);
  for (size_t candidate_index : llm_load_plan.m_selectedCandidateIndices)
  {
    if (candidate_index >= m_deviceInventory.m_candidates.size())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "LLM load plan selected invalid candidate device index {}", candidate_index);
      return false;
    }

    out_selected_devices.push_back(m_deviceInventory.m_candidates[candidate_index].m_handle);
  }

  out_selected_devices.push_back(nullptr);
  out_model_params.devices = out_selected_devices.data();
  return true;
}

bool OdaiLlamaEngine::try_load_language_model_for_plan(const std::string& base_model_path,
                                                       const std::optional<std::string>& mmproj_model_path,
                                                       const LlmLoadPlan& llm_load_plan,
                                                       LoadedLanguageModelState& out_loaded_state) const
{
  try
  {
    llama_model_params llm_model_params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> selected_devices;
    if (!build_llm_model_params(llm_load_plan, llm_model_params, selected_devices))
    {
      return false;
    }

    out_loaded_state = {};
    out_loaded_state.m_model.reset(llama_model_load_from_file(base_model_path.c_str(), llm_model_params));
    if (out_loaded_state.m_model == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to load language model with plan: {}", llm_load_plan.m_reason);
      return false;
    }

    out_loaded_state.m_vocab = llama_model_get_vocab(out_loaded_state.m_model.get());
    if (out_loaded_state.m_vocab == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to load vocabulary from language model");
      out_loaded_state = {};
      return false;
    }

    if (mmproj_model_path.has_value())
    {
      mtmd_context_params mparams = mtmd_context_params_default();
      mparams.use_gpu = false;
      mparams.print_timings = false;
      mparams.image_min_tokens = 0;
      mparams.image_max_tokens = 0;

      mtmd_context* temp_ctx = mtmd_init_from_file(mmproj_model_path->c_str(), out_loaded_state.m_model.get(), mparams);
      if (temp_ctx == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to initialize mtmd context from: {}", mmproj_model_path.value());
        out_loaded_state = {};
        return false;
      }

      out_loaded_state.m_mtmdContext = std::unique_ptr<mtmd_context, MtmdContextDeleter>(temp_ctx);
      ODAI_LOG(ODAI_LOG_INFO, "Loaded multimodal projector from {}", mmproj_model_path.value());
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while loading language model with plan '{}': {}", llm_load_plan.m_reason,
             e.what());
    out_loaded_state = {};
    return false;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught while loading language model with plan '{}'",
             llm_load_plan.m_reason);
    out_loaded_state = {};
    return false;
  }
}

std::vector<size_t> OdaiLlamaEngine::select_minimum_gpu_subset(uint64_t estimated_model_bytes,
                                                               bool& out_has_full_fit) const
{
  out_has_full_fit = false;

  std::vector<size_t> all_dgpu_indices;
  std::vector<std::pair<size_t, uint64_t>> dgpu_free_memory;

  for (size_t idx = 0; idx < m_deviceInventory.m_candidates.size(); ++idx)
  {
    const CandidateDeviceRecord& record = m_deviceInventory.m_candidates[idx];
    if (record.m_info.m_type != BackendDeviceType::GPU)
    {
      continue;
    }

    all_dgpu_indices.push_back(idx);

    const std::optional<uint64_t> free_mem = get_device_free_memory_bytes(record.m_handle);
    if (free_mem.has_value())
    {
      dgpu_free_memory.emplace_back(idx, free_mem.value());
    }
  }

  if (all_dgpu_indices.empty())
  {
    return {};
  }

  if (dgpu_free_memory.empty())
  {
    return all_dgpu_indices;
  }

  std::sort(dgpu_free_memory.begin(), dgpu_free_memory.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

  std::vector<size_t> selected_indices;
  uint64_t accumulated_free_vram = 0;
  for (const auto& [candidate_index, free_vram] : dgpu_free_memory)
  {
    selected_indices.push_back(candidate_index);
    accumulated_free_vram += free_vram;
    if (accumulated_free_vram >= estimated_model_bytes)
    {
      out_has_full_fit = true;
      break;
    }
  }

  return out_has_full_fit ? selected_indices : all_dgpu_indices;
}

OdaiLlamaEngine::LlmLoadPlan OdaiLlamaEngine::plan_llm_load(uint64_t model_file_size_bytes) const
{
  const uint64_t estimated_model_bytes = model_file_size_bytes + LLM_VRAM_OVERHEAD_BUFFER;

  // `AUTO` does not need its own branch here. Device discovery already resolved the user's
  // preference into candidate inventory, so planning only needs to inspect that inventory and
  // apply the platform-specific placement rules below.
  if (m_deviceInventory.m_requestedType == BackendDeviceType::CPU)
  {
    return make_cpu_only_llm_plan(model_file_size_bytes, "CPU-only mode was explicitly requested.");
  }

  std::optional<size_t> first_accelerator_index = std::nullopt;
  for (size_t idx = 0; idx < m_deviceInventory.m_candidates.size(); ++idx)
  {
    if (is_accelerated_device_type(m_deviceInventory.m_candidates[idx].m_info.m_type))
    {
      first_accelerator_index = idx;
      break;
    }
  }

  if (!first_accelerator_index.has_value())
  {
    // This is the soft-fallback path for inventories that ended up CPU-only after discovery.
    // We materialize that fallback explicitly instead of relying on llama.cpp defaults.
    return make_cpu_only_llm_plan(model_file_size_bytes,
                                  "No accelerator candidate was discovered, so the model will load on CPU.");
  }

#if defined(__APPLE__) || defined(__ANDROID__)
  // Phase 3 treats Apple and Android as single-accelerator / unified-memory targets:
  // once an accelerator is present, prefer full offload on that device.
  const CandidateDeviceRecord& record = m_deviceInventory.m_candidates[first_accelerator_index.value()];

  LlmLoadPlan plan;
  plan.m_mode = PlacementMode::ACCELERATED_FULL;
  plan.m_selectedCandidateIndices = {first_accelerator_index.value()};
  plan.m_shouldUseMlock = false;
  plan.m_shouldUseMmap = true;
  plan.m_nGpuLayers = N_GPU_LAYERS_ALL_POSSIBLE;
  plan.m_splitMode = LLAMA_SPLIT_MODE_NONE;
  plan.m_allowCpuRetry = true;
  plan.m_reason =
      std::format("Unified-memory platform selected accelerator '{}' for full offload.", record.m_info.m_name);
  return plan;
#else
  // Desktop policy intentionally differs by device class:
  // dGPU allows best-effort partial offload, while iGPU requires a conservative full-fit gate.
  bool has_full_dgpu_fit = false;
  const std::vector<size_t> selected_dgpu_indices = select_minimum_gpu_subset(estimated_model_bytes, has_full_dgpu_fit);
  if (!selected_dgpu_indices.empty())
  {
    LlmLoadPlan plan;
    plan.m_mode = has_full_dgpu_fit ? PlacementMode::ACCELERATED_FULL : PlacementMode::ACCELERATED_PARTIAL;
    plan.m_selectedCandidateIndices = selected_dgpu_indices;
    plan.m_shouldUseMlock = false;
    plan.m_shouldUseMmap = true;
    plan.m_nGpuLayers = N_GPU_LAYERS_ALL_POSSIBLE;
    plan.m_splitMode = selected_dgpu_indices.size() > 1 ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
    plan.m_allowCpuRetry = true;

    if (has_full_dgpu_fit)
    {
      plan.m_reason = std::format("Desktop dGPU planner selected a minimum fitting subset of {} device(s) for an "
                                  "estimated {} MB model footprint.",
                                  selected_dgpu_indices.size(), estimated_model_bytes / BYTES_PER_MB);
    }
    else
    {
      plan.m_reason = std::format("Desktop dGPU planner could not prove a full fit for the estimated {} MB model "
                                  "footprint, so it selected all discovered dGPUs for best-effort partial offload.",
                                  estimated_model_bytes / BYTES_PER_MB);
    }

    return plan;
  }

  for (size_t idx = 0; idx < m_deviceInventory.m_candidates.size(); ++idx)
  {
    const CandidateDeviceRecord& record = m_deviceInventory.m_candidates[idx];
    if (record.m_info.m_type != BackendDeviceType::IGPU)
    {
      continue;
    }

    const std::optional<uint64_t> free_memory = get_device_free_memory_bytes(record.m_handle);
    if (!free_memory.has_value())
    {
      return make_cpu_only_llm_plan(
          model_file_size_bytes,
          std::format("Desktop iGPU '{}' reported no free-memory reading, so ODAI will not risk accelerated load.",
                      record.m_info.m_name));
    }

    if (free_memory.value() < estimated_model_bytes)
    {
      return make_cpu_only_llm_plan(
          model_file_size_bytes,
          std::format("Desktop iGPU '{}' has {} MB free but the model estimate is {} MB, so ODAI will use CPU instead.",
                      record.m_info.m_name, free_memory.value() / BYTES_PER_MB, estimated_model_bytes / BYTES_PER_MB));
    }

    LlmLoadPlan plan;
    plan.m_mode = PlacementMode::ACCELERATED_FULL;
    plan.m_selectedCandidateIndices = {idx};
    plan.m_shouldUseMlock = false;
    plan.m_shouldUseMmap = true;
    plan.m_nGpuLayers = N_GPU_LAYERS_ALL_POSSIBLE;
    plan.m_splitMode = LLAMA_SPLIT_MODE_NONE;
    plan.m_allowCpuRetry = true;
    plan.m_reason = std::format(
        "Desktop iGPU '{}' passed the full-fit gate with {} MB free for an estimated {} MB model footprint.",
        record.m_info.m_name, free_memory.value() / BYTES_PER_MB, estimated_model_bytes / BYTES_PER_MB);
    return plan;
  }

  return make_cpu_only_llm_plan(model_file_size_bytes,
                                "No desktop dGPU or iGPU candidate was available for accelerated placement.");
#endif
}

OdaiResult<void> OdaiLlamaEngine::initialize_engine()
{
  try
  {
    llama_backend_init();

    OdaiResult<void> config_res = discover_candidate_devices(m_backendEngineConfig.m_preferredDeviceType);
    if (!config_res)
    {
      return config_res;
    }

    ODAI_LOG(ODAI_LOG_INFO, "Initialized llama backend and execution context");

    llama_log_set(llama_log_redirect, nullptr);
    mtmd_helper_log_set(llama_log_redirect, nullptr);

    this->m_isInitialized = true;
    return {};
  }
  ODAI_CATCH_RETURN(unexpected_internal_error<void>())
}

OdaiResult<std::vector<BackendDevice>> OdaiLlamaEngine::get_candidate_devices()
{
  if (!this->m_isInitialized)
  {
    return tl::make_unexpected(OdaiResultEnum::NOT_INITIALIZED);
  }

  std::vector<BackendDevice> candidate_devices;
  candidate_devices.reserve(m_deviceInventory.m_candidates.size());
  for (const CandidateDeviceRecord& record : m_deviceInventory.m_candidates)
  {
    candidate_devices.push_back(record.m_info);
  }

  return candidate_devices;
}

bool OdaiLlamaEngine::does_model_support_input_data(const std::vector<InputItem>& items,
                                                    const LLMModelConfig& llm_model_config, const ModelFiles& files)
{
  std::unique_ptr<mtmd_context, MtmdContextDeleter> temp_ctx = nullptr;

  if (!this->load_language_model(files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load language model");
    return false;
  }

  bool supports_image = this->m_loadedLlmState.m_mtmdContext != nullptr
                            ? mtmd_support_vision(this->m_loadedLlmState.m_mtmdContext.get())
                            : false;
  bool supports_audio = this->m_loadedLlmState.m_mtmdContext != nullptr
                            ? mtmd_support_audio(this->m_loadedLlmState.m_mtmdContext.get())
                            : false;

  for (const InputItem& item : items)
  {
    MediaType media_type = item.get_media_type();

    if (media_type == MediaType::INVALID)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for prompt");
      return false;
    }

    if (media_type == MediaType::IMAGE && !supports_image)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model does not support image input");
      return false;
    }

    if (media_type == MediaType::AUDIO && !supports_audio)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model does not support audio input");
      return false;
    }

    if (media_type == MediaType::TEXT && item.m_type != InputItemType::MEMORY_BUFFER)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Text input items must be provided as MEMORY_BUFFER");
      return false;
    }

    if ((media_type == MediaType::IMAGE || media_type == MediaType::AUDIO) && item.m_type != InputItemType::FILE_PATH)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Image / Audio input items must be provided as File Path");
      return false;
    }
  }

  return true;
}

std::unique_ptr<llama_context, LlamaContextDeleter> OdaiLlamaEngine::get_new_llama_context(ModelType model_type)
{
  std::unique_ptr<llama_context, LlamaContextDeleter> context = nullptr;
  llama_context_params context_params = llama_context_default_params();
  llama_model* model = nullptr;

  context_params.n_threads = 4;

  if (model_type == ModelType::LLM)
  {
    model = this->m_loadedLlmState.m_model.get();
    context_params.n_ctx = 2048;
    context_params.embeddings = false;
  }
  else if (model_type == ModelType::EMBEDDING)
  {
    model = this->m_embeddingModel.get();
    context_params.n_ctx = 512;
    context_params.embeddings = true;
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed");
    return nullptr;
  }

  if (model == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Model not loaded yet hence can't create context");
    return nullptr;
  }

  context.reset(llama_init_from_model(model, context_params));
  return context;
}

std::unique_ptr<llama_sampler, LlamaSamplerDeleter>
OdaiLlamaEngine::get_new_llm_llama_sampler(const SamplerConfig& config)
{
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler = nullptr;

  llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();

  sampler.reset(llama_sampler_chain_init(sampler_params));

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create llama sampler");
    return nullptr;
  }

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(config.m_topK));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(config.m_topP, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

  return sampler;
}

bool OdaiLlamaEngine::validate_model_file_entry(const std::unordered_map<std::string, std::string>& entries,
                                                const std::string& key, bool is_optional)
{
  if (!entries.contains(key))
  {
    if (is_optional)
    {
      return true;
    }
    ODAI_LOG(ODAI_LOG_ERROR, "Missing '{}' in model registration entries", key);
    return false;
  }

  const std::string& path = entries.at(key);
  if (is_optional && path.empty())
  {
    return true; // Optional file path can be empty
  }

  if (path.empty() || !std::filesystem::exists(path))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid or missing file at '{}': {}", key, path);
    return false;
  }

  return true;
}

OdaiResult<bool> OdaiLlamaEngine::validate_model_files(const ModelFiles& files)
{
  try
  {
    if (files.m_engineType != LLAMA_BACKEND_ENGINE)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid Engine Type passed");
      return false;
    }

    if (files.m_modelType == ModelType::LLM)
    {
      if (files.m_entries.size() > 2)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid number of entries passed");
        return false;
      }

      if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "base_model_path", false))
      {
        return false;
      }

      if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "mmproj_model_path", true))
      {
        return false;
      }
    }
    else if (files.m_modelType == ModelType::EMBEDDING)
    {
      if (files.m_entries.size() != 1)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Invalid number of entries passed");
        return false;
      }

      if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "base_model_path", false))
      {
        return false;
      }
    }
    else
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed");
      return false;
    }

    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while validating model files: {}", e.what());
    return unexpected_internal_error<bool>();
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught while validating model files");
    return unexpected_internal_error<bool>();
  }
}

OdaiResult<OdaiAudioTargetSpec> OdaiLlamaEngine::get_required_audio_spec(const LLMModelConfig& config,
                                                                         const ModelFiles& model_files) const
{
  (void)config;
  (void)model_files;

  if (this->m_loadedLlmState.m_mtmdContext == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "MtmD context not initialized");
    return unexpected_not_initialized<OdaiAudioTargetSpec>();
  }

  if (!mtmd_support_audio(this->m_loadedLlmState.m_mtmdContext.get()))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Loaded model does not support audio");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  int sample_rate = mtmd_get_audio_sample_rate(this->m_loadedLlmState.m_mtmdContext.get());

  if (sample_rate <= 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid audio sample rate reported by mtmd: {}", sample_rate);
    return unexpected_internal_error<OdaiAudioTargetSpec>();
  }

  return OdaiAudioTargetSpec{static_cast<uint32_t>(sample_rate), 1};
}

bool OdaiLlamaEngine::load_embedding_model(const ModelFiles& files, const EmbeddingModelConfig& config)
{
  try
  {
    if (files.m_modelType != ModelType::EMBEDDING)
    {
      return false;
    }

    std::string path = files.m_entries.at("base_model_path");

    if (this->m_embeddingModelFiles.m_entries.at("base_model_path") == path)
    {
      ODAI_LOG(ODAI_LOG_INFO, "embedding model {} is already loaded", path);
      // update config though, some other params might have changed
      this->m_embeddingModelConfig = config;
      this->m_embeddingModelFiles = files;
      return true;
    }

    llama_model_params embedding_model_params = llama_model_default_params();
    embedding_model_params.n_gpu_layers = 0; // Load entire model on CPU
    embedding_model_params.devices = nullptr;
    embedding_model_params.main_gpu = -1;
    embedding_model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    embedding_model_params.use_mlock = false;

    this->m_embeddingModel.reset(llama_model_load_from_file(path.c_str(), embedding_model_params));

    if (this->m_embeddingModel == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to load embedding model");
      return false;
    }

    this->m_embeddingModelConfig = config;
    this->m_embeddingModelFiles = files;

    ODAI_LOG(ODAI_LOG_INFO, "successfully loaded embedding model {}", path);
    return true;
  }
  catch (const std::exception& e)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Exception caught while loading embedding model: {}", e.what());
    this->m_embeddingModel.reset();
    return false;
  }
  catch (...)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unknown exception caught while loading embedding model");
    this->m_embeddingModel.reset();
    return false;
  }
}

bool OdaiLlamaEngine::load_language_model(const ModelFiles& files, const LLMModelConfig& config)
{
  if (files.m_modelType != ModelType::LLM)
  {
    return false;
  }

  if (this->m_loadedLlmState.matches(files, config))
  {
    ODAI_LOG(ODAI_LOG_INFO, "Model is already loaded with same config");
    return true;
  }

  const std::string& base_path = files.m_entries.at("base_model_path");
  std::optional<std::string> mmproj_path = std::nullopt;
  if (files.m_entries.contains("mmproj_model_path") && !files.m_entries.at("mmproj_model_path").empty())
  {
    mmproj_path = files.m_entries.at("mmproj_model_path");
  }

  ODAI_LOG(ODAI_LOG_INFO, "Preparing LLM reload transaction for {}", base_path);

  std::error_code model_size_error;
  const uint64_t model_file_size_bytes = std::filesystem::file_size(base_path, model_size_error);
  if (model_size_error)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to determine model file size for {}: {}", base_path, model_size_error.message());
    return false;
  }

  const LlmLoadPlan llm_load_plan = this->plan_llm_load(model_file_size_bytes);
  ODAI_LOG(ODAI_LOG_INFO, "LLM placement plan: {}", llm_load_plan.m_reason);

  this->m_loadedLlmState.clear();
  ODAI_LOG(ODAI_LOG_INFO, "Released previously loaded LLM state before starting reload transaction");

  LoadedLanguageModelState loaded_state;
  if (!this->try_load_language_model_for_plan(base_path, mmproj_path, llm_load_plan, loaded_state))
  {
    if (!llm_load_plan.m_allowCpuRetry || (llm_load_plan.m_mode == PlacementMode::CPU_ONLY))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "LLM reload transaction failed without a CPU retry path");
      return false;
    }

    const LlmLoadPlan cpu_retry_plan = this->make_cpu_only_llm_plan(
        model_file_size_bytes, std::format("Accelerated load failed, retrying {} on CPU. Original plan: {}", base_path,
                                           llm_load_plan.m_reason));
    ODAI_LOG(ODAI_LOG_WARN, "Accelerated LLM load failed; retrying with CPU-only plan: {}", cpu_retry_plan.m_reason);

    if (!this->try_load_language_model_for_plan(base_path, mmproj_path, cpu_retry_plan, loaded_state))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "CPU retry also failed for {}", base_path);
      return false;
    }
  }

  loaded_state.m_config = config;
  loaded_state.m_files = files;
  this->m_loadedLlmState = std::move(loaded_state);

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded language model {}", base_path);
  return true;
}

OdaiResult<std::vector<llama_token>> OdaiLlamaEngine::tokenize(const std::string& input, bool is_first,
                                                               ModelType model_type) const
{

  const llama_vocab* vocab = nullptr;

  if (model_type == EMBEDDING)
  {
    // ToDo
  }
  else if (model_type == LLM)
  {
    vocab = this->m_loadedLlmState.m_vocab;
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid Tokenization purpose passed");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  if (vocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no vocab present for tokenization");
    return unexpected_not_initialized<std::vector<llama_token>>();
  }

  // 1. Ask llama.cpp how much space we need
  int32_t n_tokens = -llama_tokenize(vocab, input.c_str(), input.length(), NULL, 0, is_first, true);

  if (n_tokens < 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize give input");
    return unexpected_internal_error<std::vector<llama_token>>();
  }

  std::vector<llama_token> tokens(n_tokens);

  // 2. Perform the tokenization
  llama_tokenize(vocab, input.c_str(), input.length(), tokens.data(), tokens.size(), is_first, true);

  ODAI_LOG(ODAI_LOG_DEBUG, "Input Tokenized successfully , total input tokens - {}", tokens.size());
  return tokens;
}

OdaiResult<std::string> OdaiLlamaEngine::detokenize(const std::vector<llama_token>& tokens) const
{
  if (this->m_loadedLlmState.m_vocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no LLM model loaded yet, so can't detokenize");
    return unexpected_not_initialized<std::string>();
  }

  std::string result;

  for (llama_token token : tokens)
  {
    char buf[128];
    int32_t n = llama_token_to_piece(this->m_loadedLlmState.m_vocab, token, buf, sizeof(buf), 0, false);

    if (n < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to detokenize give input");
      return unexpected_internal_error<std::string>();
    }

    result += std::string(buf, n);
  }

  ODAI_LOG(ODAI_LOG_DEBUG, "Input DeTokenized successfully");
  return result;
}

OdaiResult<std::string> OdaiLlamaEngine::flush_utf8_safe_output(std::vector<llama_token>& buffered_tokens,
                                                                std::string& output_buffer)
{
  OdaiResult<std::string> detokenized_res = this->detokenize(buffered_tokens);
  if (!detokenized_res)
  {
    return tl::unexpected(detokenized_res.error());
  }

  output_buffer += detokenized_res.value();

  size_t safe_len = get_safe_utf8_length(output_buffer);
  std::string safe_output_buffer = output_buffer.substr(0, safe_len);

  // updating output buffer so that it only contains remaining unsafe part
  output_buffer = output_buffer.substr(safe_len);
  // clearing buffered tokens so that its fresh next time
  buffered_tokens.clear();

  return safe_output_buffer;
}

void OdaiLlamaEngine::add_tokens_to_batch(const std::vector<llama_token>& tokens, llama_batch& batch,
                                          uint32_t& start_pos, const llama_seq_id seq_id,
                                          const bool set_logit_request_for_last_token)
{
  for (llama_token token : tokens)
  {
    int32_t i = batch.n_tokens;
    batch.token[i] = token;
    batch.pos[i] = start_pos;
    batch.n_seq_id[i] = 1;       // single sequence
    batch.seq_id[i][0] = seq_id; // only one sequence that is sequence seq_id
    batch.logits[i] = 0;         // no logits output for input tokens
    start_pos++;
    batch.n_tokens++;
  }

  batch.logits[batch.n_tokens - 1] = set_logit_request_for_last_token ? 1 : 0; // request logits for the last token
}

bool OdaiLlamaEngine::load_tokens_into_context_impl(llama_context& model_context,
                                                    const std::vector<llama_token>& tokens, uint32_t& next_pos,
                                                    const bool request_logits_for_last_token)
{
  if (tokens.empty())
  {
    ODAI_LOG(ODAI_LOG_WARN, "empty token sequence passed");
    return true;
  }

  const uint32_t n_ctx = llama_n_ctx(&model_context);
  uint32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) + 1;

  if (n_ctx_used + tokens.size() > n_ctx)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "token sequence length {} exceeds model context window (used {}/{}).", tokens.size(),
             n_ctx_used, n_ctx);
    return false;
  }

  next_pos = n_ctx_used;

  std::unique_ptr<llama_batch, LlamaBatchDeleter> batch = nullptr;

  batch.reset(new llama_batch(llama_batch_init(tokens.size(), 0, 1)));

  OdaiLlamaEngine::add_tokens_to_batch(tokens, *batch, next_pos, 0, request_logits_for_last_token);

  if (llama_decode(&model_context, *batch) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_decode failed");
    return false;
  }

  return true;
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::string& prompt, uint32_t& next_pos,
                                        const bool request_logits_for_last_token)
{
  const bool is_first = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;

  OdaiResult<std::vector<llama_token>> prompt_tokens_res = this->tokenize(prompt, is_first, ModelType::LLM);
  if (!prompt_tokens_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt, error code: {}",
             static_cast<std::uint32_t>(prompt_tokens_res.error()));
    return false;
  }

  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, prompt_tokens_res.value(), next_pos,
                                                        request_logits_for_last_token);
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::vector<llama_token>& tokens,
                                        uint32_t& next_pos, const bool request_logits_for_last_token)
{
  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, tokens, next_pos, request_logits_for_last_token);
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::string& prompt,
                                        const std::vector<mtmd::bitmap>& bitmaps, uint32_t& next_pos,
                                        bool request_logits_for_last_token)
{
  if (bitmaps.empty())
  {
    return this->load_into_context(model_context, prompt, next_pos, request_logits_for_last_token);
  }

  mtmd_input_text text;
  text.text = prompt.c_str();
  text.add_special = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;
  text.parse_special = true;

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  std::vector<const mtmd_bitmap*> bitmaps_c_ptr(bitmaps.size());
  for (size_t i = 0; i < bitmaps.size(); ++i)
  {
    bitmaps_c_ptr[i] = bitmaps[i].ptr.get();
  }

  int32_t res = mtmd_tokenize(this->m_loadedLlmState.m_mtmdContext.get(), chunks.ptr.get(), &text, bitmaps_c_ptr.data(),
                              bitmaps_c_ptr.size());
  if (res != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt with mtmd, res = {}", res);
    return false;
  }

  llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) + 1;
  n_past = std::max(n_past, 0);

  uint32_t n_batch = 512;
  llama_pos new_n_past = 0;

  if (mtmd_helper_eval_chunks(this->m_loadedLlmState.m_mtmdContext.get(), &model_context, chunks.ptr.get(), n_past, 0,
                              n_batch, request_logits_for_last_token, &new_n_past) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to evaluate chunks using mtmd in context");
    return false;
  }

  next_pos = static_cast<uint32_t>(new_n_past);
  return true;
}

bool OdaiLlamaEngine::generate_next_token(llama_context& model_context, llama_sampler& sampler, llama_token& out_token,
                                          const bool append_to_context)
{

  llama_token generated_token = llama_sampler_sample(&sampler, &model_context, -1);

  if (generated_token == LLAMA_TOKEN_NULL)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_sampler_sample failed");
    return false;
  }

  llama_sampler_accept(&sampler, generated_token);

  out_token = generated_token;

  uint32_t next_pos = 0;

  if (append_to_context)
  {
    std::vector<llama_token> token_vec = {generated_token};
    if (!load_into_context(model_context, token_vec, next_pos, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to append generated token to context");
      return false;
    }
  }

  return true;
}

OdaiResult<StreamingStats>
OdaiLlamaEngine::generate_streaming_response_impl(llama_context& model_context, llama_sampler& sampler,
                                                  const std::string& prompt, const std::vector<mtmd::bitmap>& bitmaps,
                                                  OdaiStreamRespCallbackFn callback, void* user_data)
{
  uint32_t next_pos = 0;

  if (!this->load_into_context(model_context, prompt, bitmaps, next_pos, true))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load prompt into context");
    return unexpected_internal_error<StreamingStats>();
  }

  llama_token generated_token = 0;
  std::vector<llama_token> buffered_tokens;
  int32_t total_tokens = 0;
  std::string output_buffer;

  while (true)
  {
    if (!OdaiLlamaEngine::generate_next_token(model_context, sampler, generated_token, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to generate next token");
      return unexpected_internal_error<StreamingStats>();
    }

    // Check if model is done
    if (llama_vocab_is_eog(this->m_loadedLlmState.m_vocab, generated_token))
    {
      // flush left over tokens
      if (!buffered_tokens.empty())
      {
        OdaiResult<std::string> safe_output_res = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
        if (!safe_output_res)
        {
          ODAI_LOG(ODAI_LOG_ERROR, "failed to flush buffered output, error code: {}",
                   static_cast<std::uint32_t>(safe_output_res.error()));
          return tl::unexpected(safe_output_res.error());
        }
        const std::string& safe_output_buffer = safe_output_res.value();
        if (!callback(safe_output_buffer.c_str(), user_data))
        {
          return StreamingStats{.m_generatedTokens = total_tokens, .m_wasCancelled = true};
        }
      }
      break;
    }

    buffered_tokens.push_back(generated_token);
    total_tokens++;

    if ((buffered_tokens.size() % 20) == 0)
    {
      OdaiResult<std::string> safe_output_res = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
      if (!safe_output_res)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to flush buffered output, error code: {}",
                 static_cast<std::uint32_t>(safe_output_res.error()));
        return tl::unexpected(safe_output_res.error());
      }
      const std::string& safe_output_buffer = safe_output_res.value();
      if (!callback(safe_output_buffer.c_str(), user_data))
      {
        return StreamingStats{.m_generatedTokens = total_tokens, .m_wasCancelled = true};
      }
    }
  }

  return StreamingStats{.m_generatedTokens = total_tokens, .m_wasCancelled = false};
}

OdaiResult<StreamingStats> OdaiLlamaEngine::generate_streaming_response(
    const std::vector<InputItem>& prompt, const LLMModelConfig& llm_model_config, const ModelFiles& model_files,
    const SamplerConfig& sampler_config, OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet hence can't generate response");
    return unexpected_not_initialized<StreamingStats>();
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "empty callback is passed so can't stream response");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  OdaiResult<bool> model_validation_res = validate_model_files(model_files);
  if (!model_validation_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "model file validation failed with operational error: {}",
             static_cast<std::uint32_t>(model_validation_res.error()));
    return tl::unexpected(model_validation_res.error());
  }
  if (!model_validation_res.value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid model files passed");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  if (!does_model_support_input_data(prompt, llm_model_config, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid input data");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  if (!this->load_language_model(model_files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load given language model");
    return unexpected_internal_error<StreamingStats>();
  }

  std::unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return unexpected_internal_error<StreamingStats>();
  }

  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> llm_llama_sampler =
      OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (llm_llama_sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create sampler");
    return unexpected_internal_error<StreamingStats>();
  }

  OdaiResult<std::pair<std::string, std::vector<mtmd::bitmap>>> process_result = this->process_input_items(prompt);
  if (!process_result)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to process input items, error code: {}",
             static_cast<std::uint32_t>(process_result.error()));
    return tl::unexpected(process_result.error());
  }

  std::string text_prompt = process_result->first;
  std::vector<mtmd::bitmap> bitmaps = std::move(process_result->second);

  return generate_streaming_response_impl(*llm_llama_context, *llm_llama_sampler, text_prompt, bitmaps, callback,
                                          user_data);
}

OdaiResult<std::pair<std::string, std::vector<mtmd::bitmap>>>
OdaiLlamaEngine::process_input_items(const std::vector<InputItem>& items)
{
  std::string text_content;
  std::vector<mtmd::bitmap> bitmaps;
  std::optional<OdaiAudioTargetSpec> cached_audio_spec;

  for (const InputItem& item : items)
  {
    MediaType media_type = item.get_media_type();
    if (media_type == MediaType::TEXT)
    {
      std::string part_prompt = byte_vector_to_string(item.m_data);
      text_content += part_prompt;
    }
    else if (media_type == MediaType::IMAGE)
    {
      text_content += mtmd_default_marker();
      std::string file_path = byte_vector_to_string(item.m_data);

      std::unique_ptr<IOdaiImageDecoder> image_decoder = OdaiSdk::get_new_odai_image_decoder_instance();
      if (!image_decoder)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "No image decoder available to decode image input");
        return unexpected_not_initialized<std::pair<std::string, std::vector<mtmd::bitmap>>>();
      }

      // Request original dimensions but exactly 3 channels (RGB) for mtmd
      OdaiImageTargetSpec target_spec;
      target_spec.m_maxWidth = 0;
      target_spec.m_maxHeight = 0;
      target_spec.m_channels = 3;

      OdaiDecodedImage decoded_image;
      OdaiResult<void> decode_res = image_decoder->decode_to_spec(item, target_spec, decoded_image);
      if (!decode_res)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "image decoding failed for path {} with error code {}", file_path,
                 static_cast<std::uint32_t>(decode_res.error()));
        return tl::unexpected(decode_res.error());
      }

      mtmd_bitmap* bmp = mtmd_bitmap_init(decoded_image.m_width, decoded_image.m_height, decoded_image.m_pixels.data());
      if (bmp == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to create mtmd_bitmap from decoded image");
        return unexpected_internal_error<std::pair<std::string, std::vector<mtmd::bitmap>>>();
      }
      bitmaps.emplace_back(bmp);
    }
    else if (media_type == MediaType::AUDIO)
    {
      text_content += mtmd_default_marker();
      std::unique_ptr<IOdaiAudioDecoder> audio_decoder = OdaiSdk::get_new_odai_audio_decoder_instance();
      if (!audio_decoder)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "No audio decoder available to decode audio input");
        return unexpected_not_initialized<std::pair<std::string, std::vector<mtmd::bitmap>>>();
      }

      if (!cached_audio_spec.has_value())
      {
        OdaiResult<OdaiAudioTargetSpec> spec_res =
            get_required_audio_spec(this->m_loadedLlmState.m_config, this->m_loadedLlmState.m_files);
        if (!spec_res)
        {
          ODAI_LOG(ODAI_LOG_ERROR, "Failed to get target audio spec, error code: {}",
                   static_cast<std::uint32_t>(spec_res.error()));
          return tl::unexpected(spec_res.error());
        }
        cached_audio_spec = spec_res.value();
      }
      OdaiDecodedAudio decoded_audio;
      OdaiResult<void> decode_res = audio_decoder->decode_to_spec(item, cached_audio_spec.value(), decoded_audio);
      if (!decode_res)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "audio decoding failed with error code {}",
                 static_cast<std::uint32_t>(decode_res.error()));
        return tl::unexpected(decode_res.error());
      }
      mtmd_bitmap* bmp = mtmd_bitmap_init_from_audio(decoded_audio.m_samples.size(), decoded_audio.m_samples.data());
      if (bmp == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to create mtmd_bitmap from audio");
        return unexpected_internal_error<std::pair<std::string, std::vector<mtmd::bitmap>>>();
      }
      bitmaps.emplace_back(bmp);
    }
    else
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for prompt");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }
  }

  return std::make_pair(text_content, std::move(bitmaps));
}

OdaiResult<std::string>
OdaiLlamaEngine::format_chat_messages_to_prompt(const std::vector<std::pair<std::string, std::string>>& messages,
                                                bool add_generation_prompt) const
{
  if (this->m_loadedLlmState.m_model == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no model loaded yet");
    return unexpected_not_initialized<std::string>();
  }

  // Get the chat template from the model
  const char* tmpl = llama_model_chat_template(this->m_loadedLlmState.m_model.get(), nullptr);

  if (tmpl == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat template from model");
    return unexpected_internal_error<std::string>();
  }

  ODAI_LOG(ODAI_LOG_TRACE, "Got chat template from model: {}", tmpl);

  std::vector<llama_chat_message> llama_messages;
  llama_messages.reserve(messages.size());
  for (size_t i = 0; i < messages.size(); ++i)
  {
    llama_messages.push_back({messages[i].first.c_str(), messages[i].second.c_str()});
  }

  if (messages.size() == 1 && messages[0].first == "system")
  {
    ODAI_LOG(ODAI_LOG_WARN, "Since only system message is present, appending empty user "
                            "message to avoid chat template issues where they expect at least "
                            "one user message if system message is present");
    llama_messages.push_back({"user", ""});
  }

  std::vector<char> formatted_buffer(2048);

  // Estimate needed buffer size first
  int32_t needed_size =
      llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
                                formatted_buffer.data(), formatted_buffer.size());

  if (needed_size <= 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to calculate required template buffer size");
    return unexpected_internal_error<std::string>();
  }

  if (needed_size > static_cast<int32_t>(formatted_buffer.size()))
  {
    // Allocate buffer and apply template
    formatted_buffer.resize(needed_size);
    int32_t actual_size =
        llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
                                  formatted_buffer.data(), formatted_buffer.size());
    if (actual_size <= 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to apply chat template");
      return unexpected_internal_error<std::string>();
    }
  }

  ODAI_LOG(ODAI_LOG_DEBUG, "Formatted prompt: {}", formatted_buffer.data());
  return std::string(formatted_buffer.data());
}

OdaiResult<std::unique_ptr<llama_context, LlamaContextDeleter>>
OdaiLlamaEngine::load_chat_messages_into_context(const std::vector<ChatMessage>& messages)
{

  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet");
    return unexpected_not_initialized<std::unique_ptr<llama_context, LlamaContextDeleter>>();
  }

  std::unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return unexpected_internal_error<std::unique_ptr<llama_context, LlamaContextDeleter>>();
  }

  std::vector<mtmd::bitmap> bitmaps;
  std::vector<std::pair<std::string, std::string>> extracted_messages;

  for (const ChatMessage& msg : messages)
  {
    OdaiResult<std::pair<std::string, std::vector<mtmd::bitmap>>> process_result =
        this->process_input_items(msg.m_contentItems);
    if (!process_result)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to process chat message content items, error code: {}",
               static_cast<std::uint32_t>(process_result.error()));
      return tl::unexpected(process_result.error());
    }

    extracted_messages.push_back({msg.m_role, process_result->first});

    // Move the extracted bitmaps into our main bitmaps vector
    for (auto& bmp : process_result->second)
    {
      bitmaps.emplace_back(std::move(bmp));
    }
  }

  // Format the chat messages into a prompt string (without generation prompt)
  OdaiResult<std::string> formatted_prompt_res = this->format_chat_messages_to_prompt(extracted_messages, false);
  if (!formatted_prompt_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to format chat messages into prompt, error code: {}",
             static_cast<std::uint32_t>(formatted_prompt_res.error()));
    return tl::unexpected(formatted_prompt_res.error());
  }
  const std::string& formatted_prompt = formatted_prompt_res.value();

  // Load the formatted prompt into the context to build KV cache
  uint32_t next_pos = 0;

  if (!this->load_into_context(*llm_llama_context, formatted_prompt, bitmaps, next_pos, false))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load formatted prompt into context");
    return unexpected_internal_error<std::unique_ptr<llama_context, LlamaContextDeleter>>();
  }

  ODAI_LOG(ODAI_LOG_INFO, "Successfully loaded chat context");
  return std::move(llm_llama_context);
}

OdaiResult<StreamingStats> OdaiLlamaEngine::generate_streaming_chat_response(
    const std::vector<InputItem>& prompt, const std::vector<ChatMessage>& chat_history,
    const LLMModelConfig& llm_model_config, const ModelFiles& model_files, const SamplerConfig& sampler_config,
    OdaiStreamRespCallbackFn callback, void* user_data)
{

  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet hence can't generate response");
    return unexpected_not_initialized<StreamingStats>();
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "empty callback is passed so can't stream response");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  OdaiResult<bool> model_validation_res = validate_model_files(model_files);
  if (!model_validation_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "model file validation failed with operational error: {}",
             static_cast<std::uint32_t>(model_validation_res.error()));
    return tl::unexpected(model_validation_res.error());
  }
  if (!model_validation_res.value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid model files passed");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  if (!does_model_support_input_data(prompt, llm_model_config, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid input data");
    return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
  }

  if (!this->load_language_model(model_files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load given language model");
    return unexpected_internal_error<StreamingStats>();
  }

  OdaiResult<std::unique_ptr<llama_context, LlamaContextDeleter>> chat_context_res =
      this->load_chat_messages_into_context(chat_history);
  if (!chat_context_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load chat history into cache, error code: {}",
             static_cast<std::uint32_t>(chat_context_res.error()));
    return tl::unexpected(chat_context_res.error());
  }
  std::unique_ptr<llama_context, LlamaContextDeleter> chat_context = std::move(chat_context_res.value());

  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler =
      OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create new sampler");
    return unexpected_internal_error<StreamingStats>();
  }

  OdaiResult<std::pair<std::string, std::vector<mtmd::bitmap>>> process_result = this->process_input_items(prompt);
  if (!process_result)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to process chat input items, error code: {}",
             static_cast<std::uint32_t>(process_result.error()));
    return tl::unexpected(process_result.error());
  }

  std::vector<std::pair<std::string, std::string>> extracted_msgs;
  extracted_msgs.push_back({"user", process_result->first});
  std::vector<mtmd::bitmap> bitmaps = std::move(process_result->second);

  OdaiResult<std::string> formatted_prompt_res = this->format_chat_messages_to_prompt(extracted_msgs, true);
  if (!formatted_prompt_res)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to format chat prompt, error code: {}",
             static_cast<std::uint32_t>(formatted_prompt_res.error()));
    return tl::unexpected(formatted_prompt_res.error());
  }
  std::string formatted_prompt = formatted_prompt_res.value();

  // Use the cached context and sampler to generate streaming response
  return this->generate_streaming_response_impl(*chat_context, *sampler, formatted_prompt, bitmaps, callback,
                                                user_data);
}

OdaiLlamaEngine::~OdaiLlamaEngine()
{
  llama_backend_free();
}
