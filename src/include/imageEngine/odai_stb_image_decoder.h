#pragma once

#ifdef ODAI_ENABLE_STB_IMAGE
#include "imageEngine/odai_image_decoder.h"

/// Concrete implementation of IOdaiImageDecoder using the stb_image library.
class OdaiStbImageDecoder : public IOdaiImageDecoder
{
public:
  OdaiStbImageDecoder() = default;
  ~OdaiStbImageDecoder() override = default;

  OdaiStbImageDecoder(const OdaiStbImageDecoder&) = delete;
  OdaiStbImageDecoder& operator=(const OdaiStbImageDecoder&) = delete;
  OdaiStbImageDecoder(OdaiStbImageDecoder&&) = delete;
  OdaiStbImageDecoder& operator=(OdaiStbImageDecoder&&) = delete;

  /// Checks if the given image format is supported by this decoder.
  /// Formats supported by this decoder: "png", "jpg", "jpeg", "bmp", "tga", "gif", "hdr", "pnm", "ppm", "pgm", "psd"
  /// @param format The image format extension without a leading dot (e.g. "png" or "jpg"). Matching is
  /// case-insensitive.
  /// @return true if the format is supported, false otherwise
  bool is_supported(const std::string& format) override;

  /// Decodes an image InputItem and processes it to match the target specification.
  /// @param input The InputItem containing image data.
  /// @param target_spec The required output's spec.
  /// @param decoded_image The output decoded image.
  /// @return empty expected if decoding was successful, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> do_decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                     OdaiDecodedImage& decoded_image) override;
};

#endif // ODAI_ENABLE_STB_IMAGE
