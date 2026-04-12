#include "imageEngine/odai_stb_image_decoder.h"
#include "odai_logger.h"
#include "types/odai_common_types.h"
#include "types/odai_types.h"

#include "stb_image.h"
#include "stb_image_resize2.h"

#include <algorithm>
#include <memory>

bool OdaiStbImageDecoder::is_supported(const std::string& format)
{
  std::string lower_format = format;
  std::transform(lower_format.begin(), lower_format.end(), lower_format.begin(), ::tolower);

  return (lower_format == "jpg" || lower_format == "jpeg" || lower_format == "png" || lower_format == "bmp" ||
          lower_format == "tga" || lower_format == "gif" || lower_format == "hdr" || lower_format == "pic" ||
          lower_format == "pnm");
}

OdaiResult<void> OdaiStbImageDecoder::do_decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                                        OdaiDecodedImage& decoded_image)
{
  // 1. stb_image strictly expects 'int' pointers for its output parameters.
  int raw_width = 0;
  int raw_height = 0;
  int raw_channels_in_file = 0;
  int raw_desired_channels = static_cast<int>(target_spec.m_channels);

  // 2. Decode the image from the provided source (file path or memory buffer).
  using StbiPixelsPtr = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
  StbiPixelsPtr pixels(nullptr, &stbi_image_free);
  if (input.m_type == InputItemType::FILE_PATH)
  {
    std::string file_path(input.m_data.begin(), input.m_data.end());
    pixels.reset(stbi_load(file_path.c_str(), &raw_width, &raw_height, &raw_channels_in_file, raw_desired_channels));
    if (pixels == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "stbi_load failed for path {}: {}", file_path, stbi_failure_reason());
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }
  }
  else if (input.m_type == InputItemType::MEMORY_BUFFER)
  {
    if (input.m_data.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Provided memory buffer for image decoding is empty");
      return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
    }

    pixels.reset(stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(input.m_data.data()),
                                       static_cast<int>(input.m_data.size()), &raw_width, &raw_height,
                                       &raw_channels_in_file, raw_desired_channels));
    if (pixels == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "stbi_load_from_memory failed: {}", stbi_failure_reason());
      return tl::unexpected(OdaiResultEnum::VALIDATION_FAILED);
    }
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for image decoding");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  // 3. Convert raw 'int' values to fixed-width standard integer types for consistent cross-platform calculation.
  uint32_t orig_width = static_cast<uint32_t>(raw_width);
  uint32_t orig_height = static_cast<uint32_t>(raw_height);
  uint8_t actual_channels =
      (target_spec.m_channels == 0) ? static_cast<uint8_t>(raw_channels_in_file) : target_spec.m_channels;

  // 4. Calculate scaled dimensions while strictly maintaining the original aspect ratio.
  uint32_t out_w = orig_width;
  uint32_t out_h = orig_height;

  float scale = 1.0F;
  if (target_spec.m_maxWidth > 0 && out_w > target_spec.m_maxWidth)
  {
    scale = std::min(scale, static_cast<float>(target_spec.m_maxWidth) / static_cast<float>(orig_width));
  }
  if (target_spec.m_maxHeight > 0 && out_h > target_spec.m_maxHeight)
  {
    scale = std::min(scale, static_cast<float>(target_spec.m_maxHeight) / static_cast<float>(orig_height));
  }

  out_w = static_cast<uint32_t>(static_cast<float>(orig_width) * scale);
  out_h = static_cast<uint32_t>(static_cast<float>(orig_height) * scale);

  // 5. Populate the output properties of the decoded image.
  decoded_image.m_width = out_w;
  decoded_image.m_height = out_h;
  decoded_image.m_channels = actual_channels;

  bool resize_required = (out_w != orig_width || out_h != orig_height);
  try
  {
    if (resize_required)
    {
      // 6a. If scaling is necessary, explicitly allocate the buffer and invoke STB resize.
      size_t total_size =
          static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * static_cast<size_t>(actual_channels);
      decoded_image.m_pixels.resize(total_size);

      // Invoke the linear resizing algorithm provided by stb_image_resize2.
      // We safely cast dimensions back to int for STB's internal use.
      auto* resized_ptr = stbir_resize_uint8_linear(
          pixels.get(), static_cast<int>(orig_width), static_cast<int>(orig_height), 0, decoded_image.m_pixels.data(),
          static_cast<int>(out_w), static_cast<int>(out_h), 0, static_cast<stbir_pixel_layout>(actual_channels));

      if (resized_ptr == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "stbir_resize_uint8_linear failed to resize the image");
        return unexpected_internal_error<void>();
      }
    }
    else
    {
      // 6b. If scaling is skipped, simply copy the original raw pixels into the output buffer.
      size_t total_size =
          static_cast<size_t>(orig_width) * static_cast<size_t>(orig_height) * static_cast<size_t>(actual_channels);
      decoded_image.m_pixels.resize(total_size);
      std::memcpy(decoded_image.m_pixels.data(), pixels.get(), total_size);
    }

    return {};
  }
  catch (const std::bad_alloc&)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to allocate memory buffer for the decoded output image");
    return unexpected_internal_error<void>();
  }
}
