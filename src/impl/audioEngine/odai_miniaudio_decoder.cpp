#include "audioEngine/odai_miniaudio_decoder.h"
#include "miniaudio.h"
#include "odai_sdk.h"

// Helper function to handle reading from either an initialized from-file or from-memory decoder
static bool read_pcm_from_decoder(ma_decoder& decoder, OdaiDecodedAudio& decoded_audio)
{
  ma_uint64 frame_count = 0;
  ma_result result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
  if (result != MA_SUCCESS)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get PCM frame length from decoder");
    return false;
  }

  decoded_audio.m_sampleRate = decoder.outputSampleRate;
  decoded_audio.m_channels = decoder.outputChannels;
  decoded_audio.m_samples.resize(frame_count * decoder.outputChannels);

  ma_uint64 frames_read = 0;
  result = ma_decoder_read_pcm_frames(&decoder, decoded_audio.m_samples.data(), frame_count, &frames_read);
  if (result != MA_SUCCESS || frames_read == 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to read PCM frames from decoder");
    return false;
  }

  if (frames_read < frame_count)
  {
    ODAI_LOG(ODAI_LOG_WARN, "Read fewer frames than expected from decoder");
  }

  return true;
}

bool OdaiMiniAudioDecoder::decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                                          OdaiDecodedAudio& decoded_audio)
{
  ma_decoder_config decoder_config =
      ma_decoder_config_init(ma_format_f32, target_spec.m_channels, target_spec.m_sampleRate);
  ma_decoder decoder;

  ma_result result{};

  if (input.m_type == InputItemType::FILE_PATH)
  {
    std::string file_path(input.m_data.begin(), input.m_data.end());
    result = ma_decoder_init_file(file_path.c_str(), &decoder_config, &decoder);
    if (result != MA_SUCCESS)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize miniaudio decoder for file: {}", file_path);
      return false;
    }
  }
  else if (input.m_type == InputItemType::MEMORY_BUFFER)
  {
    if (input.m_data.empty())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Provided memory buffer for audio decoding is empty");
      return false;
    }
    result = ma_decoder_init_memory(input.m_data.data(), input.m_data.size(), &decoder_config, &decoder);
    if (result != MA_SUCCESS)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize miniaudio decoder from memory buffer");
      return false;
    }
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for audio decoding");
    return false;
  }

  bool read_status = read_pcm_from_decoder(decoder, decoded_audio);
  ma_decoder_uninit(&decoder);
  return read_status;
}