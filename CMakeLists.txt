cmake_minimum_required(VERSION 3.10)

# -----------------------------------------------------------------
# 1. Project & Global Config
# -----------------------------------------------------------------
project(odai C CXX)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

# Globally disable versioned filenames (libfoo.so.1) and symlinks for ALL shared libraries
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT iporesult)

# Determine Platform & Optimization
if(ANDROID)
    set(TARGET_PLATFORM "android")
    add_compile_definitions(__ANDROID__)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        add_compile_options(-march=armv8.2-a+dotprod)
    endif()
elseif(APPLE)
        set(TARGET_PLATFORM "apple")
        add_compile_definitions(__APPLE__)
elseif(WIN32)
    # WIN32 covers all Windows versions (32-bit and 64-bit).
    set(TARGET_PLATFORM "windows")
    add_compile_definitions(__WINDOWS__)
elseif(UNIX)
    set(TARGET_PLATFORM "linux")
    add_compile_definitions(__LINUX__)
else()
    set(TARGET_PLATFORM "generic")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3) 
endif()

message(STATUS ">>> Configuring for Platform: ${TARGET_PLATFORM} (${CMAKE_BUILD_TYPE})")

# Organize Output Directories
# Files will go to: PROJECT_ROOT/out/Android/Debug/ or out/Linux/Release/
set(OUTPUT_BASE_DIR "${CMAKE_SOURCE_DIR}/out/${TARGET_PLATFORM}/${CMAKE_BUILD_TYPE}")
if(ANDROID)
    set(OUTPUT_BASE_DIR "${OUTPUT_BASE_DIR}/${ANDROID_ABI}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_BASE_DIR}/archives")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_BASE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_BASE_DIR}/bin")

# -----------------------------------------------------------------
# 2. Dependencies (FetchContent)
# -----------------------------------------------------------------
include(FetchContent)

# Function to configure if third-party dependencies should be built as shared/static.
# This is preferred over global BUILD_SHARED_LIBS to maintain control over dependency linkage.
function(odai_configure_dependency_build_type name type)
    if(type STREQUAL "SHARED")
        set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    else()
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
endfunction()

# --- nlohmann/json ---
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0)
odai_configure_dependency_build_type(nlohmann_json STATIC)
FetchContent_MakeAvailable(nlohmann_json)

# --- xxHash ---
FetchContent_Declare(xxHash GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git GIT_TAG v0.8.3)
odai_configure_dependency_build_type(xxHash STATIC)
FetchContent_MakeAvailable(xxHash)

# --- tl::expected ---
FetchContent_Declare(tl_expected GIT_REPOSITORY https://github.com/TartanLlama/expected.git GIT_TAG v1.3.1)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
odai_configure_dependency_build_type(tl_expected STATIC)
FetchContent_MakeAvailable(tl_expected)

# --- SQLiteCpp ---
if(ODAI_ENABLE_SQLITE_DB)
    FetchContent_Declare(SQLiteCpp GIT_REPOSITORY https://github.com/SRombauts/SQLiteCpp.git GIT_TAG 3.3.3)
    set(SQLITECPP_INTERNAL_SQLITE ON CACHE BOOL "" FORCE)
    set(SQLITECPP_RUN_CPPLINT OFF CACHE BOOL "" FORCE)
    odai_configure_dependency_build_type(SQLiteCpp STATIC)
    FetchContent_MakeAvailable(SQLiteCpp)

    add_library(sqlite-vec STATIC sqlite-vec-0.1.6-amalgamation/sqlite-vec.c)
    target_include_directories(sqlite-vec PRIVATE sqlite-vec-0.1.6-amalgamation)
    target_link_libraries(sqlite-vec PRIVATE sqlite3)
endif()

# --- Llama.cpp ---
if(ODAI_ENABLE_LLAMA_BACKEND)
    FetchContent_Declare(llama GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git GIT_TAG b8748)

    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

    if(ODAI_CI_BUILD)
        set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    else()
        set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
    endif()
    
    # KleidiAI Optimization
    set(KLEIDIAI_ENABLE OFF)
    if(ANDROID OR APPLE)
        set(KLEIDIAI_ENABLE ON)
    # KleidiAI Optimization for Linux / Windows ARM64 laptops
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8-a")
        set(KLEIDIAI_ENABLE ON)
    endif()
    set(GGML_CPU_KLEIDIAI ${KLEIDIAI_ENABLE} CACHE BOOL "" FORCE)

    # GPU Backend Auto-detection
    set(METAL_DEFAULT OFF)
    set(CUDA_DEFAULT OFF)
    set(HIP_DEFAULT OFF)
    set(SYCL_DEFAULT OFF)
    set(VULKAN_DEFAULT OFF)

    if(APPLE)
        set(METAL_DEFAULT ON)
    elseif(ANDROID)
        set(VULKAN_DEFAULT ON)
    elseif(WIN32 OR (UNIX AND NOT APPLE AND NOT ANDROID))
        if(ODAI_CI_BUILD)
            set(CUDA_DEFAULT ON)
            set(HIP_DEFAULT ON)
            set(SYCL_DEFAULT ON)
            set(VULKAN_DEFAULT ON)
        else()
        # Local Build: Auto-detection to provide a smooth experience without config errors.
        
        # Detect CUDA
            if(EXISTS "$ENV{CUDA_PATH}" OR EXISTS "/usr/local/cuda" OR EXISTS "$ENV{CUDA_HOME}")
                set(CUDA_DEFAULT ON)
            message(STATUS ">>> CUDA SDK detected, enabling CUDA backend by default")
            endif()

        # Detect HIP/ROCm
        # CMake's HIP detection fails FATALLY if enabled but not found. We only default to ON if paths exist.
            if(EXISTS "$ENV{ROCM_PATH}" OR EXISTS "/opt/rocm" OR EXISTS "/usr/local/rocm")
                set(HIP_DEFAULT ON)
            message(STATUS ">>> ROCm/HIP SDK detected (via path), enabling HIP backend by default")
            endif()

        # Detect Vulkan
            find_package(Vulkan QUIET)
            if(Vulkan_FOUND)
                set(VULKAN_DEFAULT ON)
            message(STATUS ">>> Vulkan SDK detected, enabling Vulkan backend by default")
            endif()
        endif()
    endif()

    set(GGML_CUDA ${CUDA_DEFAULT} CACHE BOOL "" FORCE)
    set(GGML_HIP ${HIP_DEFAULT} CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${VULKAN_DEFAULT} CACHE BOOL "" FORCE)
    set(GGML_METAL ${METAL_DEFAULT} CACHE BOOL "" FORCE)
    set(GGML_SYCL ${SYCL_DEFAULT} CACHE BOOL "" FORCE)

    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
    
    # llama.cpp is built as SHARED to support dynamic loading of backends
    odai_configure_dependency_build_type(llama SHARED)
    FetchContent_MakeAvailable(llama)

    # Ensure all backend targets use correct output directories
    set(LLAMA_LIBS llama ggml ggml-base mtmd)
    if(GGML_AVAILABLE_BACKENDS)
        list(APPEND LLAMA_LIBS ${GGML_AVAILABLE_BACKENDS})
    endif()

    foreach(lib ${LLAMA_LIBS})
        if(TARGET ${lib})
            set_target_properties(${lib} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
                ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
                INTERPROCEDURAL_OPTIMIZATION ${iporesult}
            )
        endif()
    endforeach()
endif()

# --- Miniaudio ---
if(ODAI_ENABLE_MINIAUDIO)
    FetchContent_Declare(miniaudio GIT_REPOSITORY https://github.com/mackron/miniaudio.git GIT_TAG 0.11.25)
    set(MINIAUDIO_NO_LIBVORBIS ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_LIBOPUS ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_EXTRA_NODES ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_DEVICEIO ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_RESOURCE_MANAGER ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_NODE_GRAPH ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_ENGINE ON CACHE BOOL "" FORCE)
    set(MINIAUDIO_NO_GENERATION ON CACHE BOOL "" FORCE)
    odai_configure_dependency_build_type(miniaudio STATIC)
    FetchContent_MakeAvailable(miniaudio)
endif()

# --- STB ---
if(ODAI_ENABLE_STB_IMAGE)
    FetchContent_Declare(stb GIT_REPOSITORY https://github.com/nothings/stb.git GIT_TAG master)
    odai_configure_dependency_build_type(stb STATIC)
    FetchContent_MakeAvailable(stb)
endif()

# -----------------------------------------------------------------
# 3. Main Library (odai)
# -----------------------------------------------------------------
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

set(ODAI_SOURCES
    src/impl/odai_logger.cpp
    src/impl/odai_public.cpp
    src/impl/odai_sdk.cpp
    src/impl/types/odai_type_conversions.cpp
    src/impl/utils/odai_helpers.cpp
    src/impl/utils/string_utils.cpp
    src/impl/ragEngine/odai_rag_engine.cpp
    src/impl/audioEngine/odai_audio_decoder.cpp
    src/impl/imageEngine/odai_image_decoder.cpp
)

add_library(odai SHARED ${ODAI_SOURCES})

target_include_directories(odai PUBLIC src/include)
target_include_directories(odai PRIVATE 
    ${nlohmann_json_SOURCE_DIR}
    ${xxHash_SOURCE_DIR}
    ${tl_expected_SOURCE_DIR}/include
)

target_compile_definitions(odai PRIVATE 
    XXH_INLINE_ALL
)

# Core linking
target_link_libraries(odai PRIVATE nlohmann_json)

# Note: xxHash and tl::expected are header-only libraries used via private includes; 
# they are not linked as separate library targets as they are integrated directly into odai.

if(iporesult)
    set_target_properties(odai PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(nlohmann_json PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Component Integrations
if(ODAI_ENABLE_LLAMA_BACKEND)
    target_sources(odai PRIVATE 
        src/impl/backendEngine/odai_llamacpp/odai_llama_type_conversions.cpp
        src/impl/backendEngine/odai_llamacpp/odai_llama_backend_engine.cpp
    )
    target_include_directories(odai PRIVATE 
        ${llama_SOURCE_DIR}
        ${llama_SOURCE_DIR}/tools/mtmd
    )
    target_link_libraries(odai PRIVATE llama mtmd)
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_LLAMA_BACKEND)
endif()

if(ODAI_ENABLE_SQLITE_DB)
    target_sources(odai PRIVATE src/impl/db/odai_sqlite/odai_sqlite_db.cpp)
    target_include_directories(odai PRIVATE ${SQLiteCpp_SOURCE_DIR})
    target_link_libraries(odai PRIVATE sqlite3 sqlite-vec SQLiteCpp)
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_SQLITE_DB)
    if(iporesult)
        set_target_properties(sqlite3 PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(sqlite-vec PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(SQLiteCpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if(ODAI_ENABLE_MINIAUDIO)
    target_sources(odai PRIVATE
        src/impl/audioEngine/odai_miniaudio_decoder.cpp
        src/impl/headerOnlyLib/odai_miniaudio_impl.cpp
    )
    target_include_directories(odai PRIVATE ${miniaudio_SOURCE_DIR})
    target_compile_definitions(odai PRIVATE 
        ODAI_ENABLE_MINIAUDIO
        MA_NO_VORBIS MA_NO_OPUS MA_NO_DEVICE_IO MA_NO_RESOURCE_MANAGER
        MA_NO_NODE_GRAPH MA_NO_ENGINE MA_NO_GENERATION
    )
endif()

if(ODAI_ENABLE_STB_IMAGE)
    target_sources(odai PRIVATE
        src/impl/imageEngine/odai_stb_image_decoder.cpp
        src/impl/headerOnlyLib/odai_stb_image_impl.cpp
    )
    target_include_directories(odai PRIVATE ${stb_SOURCE_DIR})
    # stb is header only, no target linking needed
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_STB_IMAGE)
endif()

# -----------------------------------------------------------------
# 4. Platform Specifics
# -----------------------------------------------------------------
if(ANDROID)
    add_library(odaiAndroid SHARED src/impl/odai_jni.cpp)
    target_include_directories(odaiAndroid PUBLIC src/include)
    target_link_libraries(odaiAndroid PRIVATE odai)
endif()
