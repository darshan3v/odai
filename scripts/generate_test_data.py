#!/usr/bin/env python3
"""Generate synthetic test assets for decoder integration tests.

Creates minimal image and audio files with known, deterministic properties
so tests can verify structural correctness (dimensions, channels, sample rates)
without relying on external or pre-committed binary files.

Usage:
    python generate_test_data.py <output_dir>

Requirements (installed automatically by CMake into a build-local venv):
    Pillow, pydub, imageio, opencv-python-headless, psd-tools, numpy
    ffmpeg must be available on PATH for FLAC export.

The script creates the following directory structure under <output_dir>:
    images/
        tiny_rgba.png      - 4x3 RGBA PNG (4 channels)
        tiny_rgb.bmp       - 4x3 RGB BMP (3 channels)
        tiny_rgb.gif       - 4x3 RGB GIF (3 channels)
        tiny_rgb.tga       - 4x3 RGB TGA (3 channels)
        tiny_rgb.ppm       - 4x3 RGB PPM (3 channels)
        tiny_rgb.pnm       - 4x3 RGB PNM (3 channels)
        tiny_gray.pgm      - 4x3 grayscale PGM (1 channel)
        tiny_rgb.psd       - 4x3 RGB PSD (3 channels)
        tiny_rgb.hdr       - 4x3 RGB Radiance HDR (3 channels)
    audio/
        tiny_stereo_44100.wav  - 50 ms stereo WAV at 44100 Hz
        tiny_stereo_44100.flac - 50 ms stereo FLAC at 44100 Hz
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from psd_tools import PSDImage
from pydub import AudioSegment
from pydub.generators import Sine

ImagePixels = list[list[tuple[int, ...] | int]]

# 4x3 grid of RGBA pixels. Values are deliberately varied to exercise channel
# conversion and resize logic.
RGBA_PIXELS = [
    [(255, 0, 0, 255), (0, 255, 0, 192), (0, 0, 255, 128), (255, 255, 0, 64)],
    [(255, 0, 255, 255), (0, 255, 255, 192), (255, 128, 0, 128), (64, 64, 64, 64)],
    [(255, 255, 255, 255), (0, 0, 0, 192), (128, 0, 255, 128), (0, 128, 255, 64)],
]

# Same layout as RGBA without alpha.
RGB_PIXELS = [
    [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
    [(255, 0, 255), (0, 255, 255), (255, 128, 0), (64, 64, 64)],
    [(255, 255, 255), (0, 0, 0), (128, 0, 255), (0, 128, 255)],
]

GRAY_PIXELS = [
    [76, 150, 29, 226],
    [105, 179, 151, 64],
    [255, 0, 67, 104],
]


def generate_image(path: Path, pixels: ImagePixels, **kwargs) -> None:
    """Create an image file from a pixel grid using imageio.

    Args:
        path: Output file path. Format is inferred from the extension.
        pixels: 2D list of pixel tuples or grayscale values.
        kwargs: Extra arguments forwarded to imageio.
    """
    arr = np.array(pixels, dtype=np.uint8)
    iio.imwrite(path, arr, **kwargs)


def generate_psd(path: Path, pixels: ImagePixels) -> None:
    """Generate an Adobe Photoshop PSD image using psd-tools."""
    arr = np.array(pixels, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    psd = PSDImage.frompil(img)
    psd.save(path)


def generate_audio(
    path: Path,
    *,
    sample_rate: int = 44_100,
    channels: int = 2,
    duration_ms: int = 50,
    frequencies: tuple[float, ...] = (440.0, 660.0),
    audio_format: str = "wav",
) -> None:
    """Generate an audio file with sine wave tones using pydub.

    Args:
        path: Output file path.
        sample_rate: Samples per second.
        channels: Number of audio channels (1 or 2).
        duration_ms: Length of the audio clip in milliseconds.
        frequencies: One frequency per channel. If fewer frequencies than
                     channels, the last frequency is repeated.
        audio_format: Output format, such as "wav", "mp3", or "flac".
    """
    freqs = list(frequencies)
    while len(freqs) < channels:
        freqs.append(freqs[-1])

    mono_segments = [
        Sine(freqs[ch]).to_audio_segment(duration=duration_ms).set_sample_width(2).set_frame_rate(sample_rate)
        for ch in range(channels)
    ]

    if channels == 1:
        audio = mono_segments[0]
    else:
        audio = AudioSegment.from_mono_audiosegments(*mono_segments)

    audio.export(str(path), format=audio_format)


def generate_images(image_dir: Path) -> None:
    """Generate all synthetic image fixtures."""
    generate_image(image_dir / "tiny_rgba.png", RGBA_PIXELS)
    generate_image(image_dir / "tiny_rgb.bmp", RGB_PIXELS)
    generate_image(image_dir / "tiny_rgb.gif", RGB_PIXELS)
    generate_image(image_dir / "tiny_rgb.tga", RGB_PIXELS)
    generate_image(image_dir / "tiny_rgb.ppm", RGB_PIXELS)
    generate_image(image_dir / "tiny_rgb.pnm", RGB_PIXELS, extension=".pnm")
    generate_image(image_dir / "tiny_gray.pgm", GRAY_PIXELS)
    generate_psd(image_dir / "tiny_rgb.psd", RGB_PIXELS)
    generate_image(image_dir / "tiny_rgb.hdr", RGB_PIXELS, extension=".hdr")


def generate_audio_fixtures(audio_dir: Path) -> None:
    """Generate all synthetic audio fixtures."""
    shared_audio_args = {
        "sample_rate": 44_100,
        "channels": 2,
        "duration_ms": 50,
        "frequencies": (440.0, 660.0),
    }

    generate_audio(audio_dir / "tiny_stereo_44100.wav", **shared_audio_args)
    generate_audio(audio_dir / "tiny_stereo_44100.flac", **shared_audio_args, audio_format="flac")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    image_dir = output_dir / "images"
    audio_dir = output_dir / "audio"
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    generate_images(image_dir)
    generate_audio_fixtures(audio_dir)

    print(f"Generated test data in {output_dir}")


if __name__ == "__main__":
    main()
