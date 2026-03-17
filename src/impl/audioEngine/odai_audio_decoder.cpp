#include "audioEngine/odai_audio_decoder.h"
#include "odai_logger.h"
#include "types/odai_types.h"

bool IOdaiAudioDecoder::decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                                       OdaiDecodedAudio& decoded_audio)
{
  if (!input.is_sane())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not sane for decoding");
    return false;
  }

  if (input.get_media_type() != MediaType::AUDIO)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input item is not an audio file");
    return false;
  }

  if (input.m_data.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Input audio data is empty");
    return false;
  }

  return do_decode_to_spec(input, target_spec, decoded_audio);
}
