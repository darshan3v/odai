#include "audioEngine/odai_miniaudio_decoder.h"
#include "miniaudio.h"
#include "odai_sdk.h"

bool OdaiMiniAudioDecoder::decode_to_spec(const std::string& file_path, const OdaiAudioTargetSpec& target_spec,
                                          OdaiDecodedAudio& decoded_audio)
{
  ma_decoder_config decoder_config =
      ma_decoder_config_init(ma_format_f32, target_spec.m_channels, target_spec.m_sampleRate);
  ma_decoder decoder;

  // Initialize decoder from file
  ma_result result = ma_decoder_init_file(file_path.c_str(), &decoder_config, &decoder);
  if (result != MA_SUCCESS)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to initialize miniaudio decoder for file: {}", file_path);
    return false;
  }

  ma_uint64 frame_count = 0;
  result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
  if (result != MA_SUCCESS)
  {
    ma_decoder_uninit(&decoder);
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to get PCM frame length for file: {}", file_path);
    return false;
  }

  decoded_audio.m_sampleRate = decoder.outputSampleRate;
  decoded_audio.m_channels = decoder.outputChannels;
  decoded_audio.m_samples.resize(frame_count * decoder.outputChannels);

  ma_uint64 frames_read = 0;
  result = ma_decoder_read_pcm_frames(&decoder, decoded_audio.m_samples.data(), frame_count, &frames_read);
  if (result != MA_SUCCESS || frames_read == 0)
  {
    ma_decoder_uninit(&decoder);
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to read PCM frames from file: {}", file_path);
    return false;
  }

  if (frames_read < frame_count)
  {
    ODAI_LOG(ODAI_LOG_WARN, "Read fewer frames than expected from file: {}", file_path);
  }

  ma_decoder_uninit(&decoder);
  return true;
}