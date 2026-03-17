#include "imageEngine/odai_image_decoder.h"
#include "odai_logger.h"
#include "types/odai_types.h"

bool IOdaiImageDecoder::decode_to_spec(const InputItem& input, const OdaiImageTargetSpec& target_spec,
                                       OdaiDecodedImage& decoded_image)
{
  if (!input.is_sane())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not sane for decoding");
    return false;
  }

  if (input.get_media_type() != MediaType::IMAGE)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not an image file");
    return false;
  }

  if (input.m_data.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input image data is empty");
    return false;
  }

  return do_decode_to_spec(input, target_spec, decoded_image);
}
