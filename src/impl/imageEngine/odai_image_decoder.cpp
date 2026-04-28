#include "imageEngine/odai_image_decoder.h"
#include "imageEngine/odai_stb_image_decoder.h"
#include "odai_logger.h"
#include "types/odai_types.h"

#include <memory>

std::unique_ptr<IOdaiImageDecoder> IOdaiImageDecoder::create_default()
{
#ifdef ODAI_ENABLE_STB_IMAGE
  return std::make_unique<OdaiStbImageDecoder>();
#else
  return nullptr;
#endif
}

OdaiResult<void> IOdaiImageDecoder::decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                                   OdaiDecodedImage& decoded_image)
{
  if (!input.is_sane())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not sane for decoding");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  if (input.get_media_type() != MediaType::IMAGE)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not an image file");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  if (input.m_data.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input image data is empty");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  return do_decode_to_spec(input, target_spec, decoded_image);
}
