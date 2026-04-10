#pragma once

#include "types/odai_result.h"

#include <string>

class InputItem;
struct OdaiImageTargetSpec;
struct OdaiDecodedImage;

/// Pure virtual interface for decoding image files.
class IOdaiImageDecoder
{
public:
  virtual ~IOdaiImageDecoder() = default;
  IOdaiImageDecoder() = default;

  IOdaiImageDecoder(const IOdaiImageDecoder&) = delete;
  IOdaiImageDecoder& operator=(const IOdaiImageDecoder&) = delete;
  IOdaiImageDecoder(IOdaiImageDecoder&&) = delete;
  IOdaiImageDecoder& operator=(IOdaiImageDecoder&&) = delete;

  /// Checks if the given image format is supported by this decoder.
  /// @param format The format to check (e.g. extension like "png" or "jpg")
  /// @return true if the format is supported, false otherwise
  virtual bool is_supported(const std::string& format) = 0;

  /// Decodes an image InputItem and processes it to match the target specification.
  /// @param input The InputItem containing image data.
  /// @param target_spec The required output's spec.
  /// @param decoded_image The output decoded image.
  /// @return empty expected if decoding was successful, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                  OdaiDecodedImage& decoded_image);

protected:
  /// Core implementation of decode_to_spec to be provided by the derived class.
  /// Input validation (sanity constraints, media type, empty data) is already handled by the base class.
  /// @param input The validated InputItem containing image data.
  /// @param target_spec The required output's spec.
  /// @param decoded_image The output decoded image.
  /// @return empty expected if decoding was successful, or an unexpected OdaiResultEnum indicating the error.
  virtual OdaiResult<void> do_decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                             OdaiDecodedImage& decoded_image) = 0;
};
