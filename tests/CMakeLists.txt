find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(FFMPEG_EXECUTABLE ffmpeg REQUIRED)

# ---------------------------------------------------------------------------
# Test data: venv setup + generation + sample copying (one-time, cached)
# ---------------------------------------------------------------------------
set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/.venv")
set(VENV_PYTHON "${VENV_DIR}/bin/python")
set(TEST_DATA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
set(TEST_DATA_OUTPUT_FILES
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgba.png"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.bmp"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.gif"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.tga"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.ppm"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.pnm"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_gray.pgm"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.psd"
    "${TEST_DATA_OUTPUT_DIR}/images/tiny_rgb.hdr"
    "${TEST_DATA_OUTPUT_DIR}/images/sample_chamaleon.jpg"
    "${TEST_DATA_OUTPUT_DIR}/audio/tiny_stereo_44100.wav"
    "${TEST_DATA_OUTPUT_DIR}/audio/tiny_stereo_44100.flac"
    "${TEST_DATA_OUTPUT_DIR}/audio/Echoes_of_Unseen_Light.mp3"
)

add_custom_command(
    OUTPUT
        ${TEST_DATA_OUTPUT_FILES}
    # 1. Create venv (no-op if already exists)
    COMMAND Python3::Interpreter -m venv "${VENV_DIR}"
    # 2. Install deps (instant if already installed)
    COMMAND "${VENV_DIR}/bin/pip" install --quiet Pillow pydub audioop-lts imageio opencv-python-headless psd-tools numpy
    # 3. Remove stale generated/copied assets before recreating the build-local fixture directory
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${TEST_DATA_OUTPUT_DIR}"
    # 4. Generate synthetic assets
    COMMAND "${VENV_PYTHON}"
        "${CMAKE_SOURCE_DIR}/scripts/generate_test_data.py"
        "${TEST_DATA_OUTPUT_DIR}"
    # 5. Copy real sample files alongside the generated ones
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/data/images/sample_chamaleon.jpg"
        "${TEST_DATA_OUTPUT_DIR}/images/sample_chamaleon.jpg"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/data/audio/Echoes_of_Unseen_Light.mp3"
        "${TEST_DATA_OUTPUT_DIR}/audio/Echoes_of_Unseen_Light.mp3"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/scripts/generate_test_data.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/images/sample_chamaleon.jpg"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/audio/Echoes_of_Unseen_Light.mp3"
    COMMENT "Preparing test data (venv + generate + copy samples)"
)

add_custom_target(odai_test_data DEPENDS ${TEST_DATA_OUTPUT_FILES})

# ---------------------------------------------------------------------------
# Layer test subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(db)
add_subdirectory(imageEngine)
add_subdirectory(audioEngine)
