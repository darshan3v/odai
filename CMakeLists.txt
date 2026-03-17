cmake_minimum_required(VERSION 3.10)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

# Globally disable versioned filenames (libfoo.so.1) and symlinks for ALL shared libraries
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# Project Name
project(odai C CXX)

include(CheckIPOSupported)
check_ipo_supported(RESULT iporesult)

# Determine Platform (for output naming)
if(ANDROID)
    set(TARGET_PLATFORM "android")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__android__")
elseif(APPLE)
    set(TARGET_PLATFORM "apple")
elseif(UNIX)
    set(TARGET_PLATFORM "linux")
else()  
    set(TARGET_PLATFORM "generic")
endif()

# --- Size Optimization (Critical for On-Device RAG) ---
# -Oz: Aggressive size optimization (Clang specific)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3) 
endif()

message(STATUS ">>> Configuring for Platform: ${TARGET_PLATFORM}")  
message(STATUS ">>> Build Type: ${CMAKE_BUILD_TYPE}")

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
# 1. Global Settings
# -----------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------
# 2. Build Options
# -----------------------------------------------------------------
option(ODAI_ENABLE_LLAMA_BACKEND "Enable Llama.cpp backend engine" ON)
option(ODAI_ENABLE_SQLITE_DB "Enable SQLite Database backend" ON)
option(ODAI_ENABLE_MINIAUDIO "Enable Miniaudio Audio decoding engine" ON)
option(ODAI_ENABLE_STB_IMAGE "Enable stb_image Image decoding engine" ON)

# -----------------------------------------------------------------
# 3. Core Dependencies (Always included)
# -----------------------------------------------------------------
include(FetchContent)

# --- Setup nlohmann/json ---
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

# --- Setup xxHash ---
FetchContent_Declare(
    xxHash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.2
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xxHash)

# -----------------------------------------------------------------
# 4. Optional Dependencies
# -----------------------------------------------------------------

if(ODAI_ENABLE_LLAMA_BACKEND)
    message(STATUS ">>> Enabling Llama.cpp backend")
FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        b7656 
)

# Disable llama.cpp's internal examples/tests to speed up build
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE_DEFAULT OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

# Build mtmd (multimodal) library from llama.cpp tools.
# Since LLAMA_BUILD_TOOLS is OFF, we explicitly add this subdirectory.
# LLAMA_INSTALL_VERSION is needed by mtmd's CMakeLists but is normally set by llama's root CMake.
if(NOT DEFINED LLAMA_INSTALL_VERSION)
    set(LLAMA_INSTALL_VERSION "0.0.0")
endif()
add_subdirectory(${llama_SOURCE_DIR}/tools/mtmd ${llama_BINARY_DIR}/tools/mtmd)

# Exclude CLI executables from being built
if(TARGET llama-mtmd-cli)
    set_target_properties(llama-mtmd-cli PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

set_target_properties(llama PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
    if(iporesult)
        set_target_properties(llama PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if(ODAI_ENABLE_SQLITE_DB)
    message(STATUS ">>> Enabling SQLite Database backend")
    
FetchContent_Declare(
    SQLiteCpp
    GIT_REPOSITORY https://github.com/SRombauts/SQLiteCpp.git
    GIT_TAG        3.3.3 
)

    # Tell SQLiteCpp to build its internal sqlite3
set(SQLITECPP_INTERNAL_SQLITE ON CACHE BOOL "" FORCE)
set(SQLITECPP_RUN_CPPLINT OFF CACHE BOOL "" FORCE) 

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(SQLiteCpp)

    # Setup SQLite-Vec
add_library(sqlite-vec STATIC sqlite-vec-0.1.6-amalgamation/sqlite-vec.c)
target_include_directories(sqlite-vec PRIVATE sqlite-vec-0.1.6-amalgamation)
target_link_libraries(sqlite-vec PRIVATE sqlite3)

    if(iporesult)
        set_target_properties(sqlite3 PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(sqlite-vec PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(SQLiteCpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

if(ODAI_ENABLE_MINIAUDIO)
    message(STATUS ">>> Enabling Miniaudio Audio decoder")
FetchContent_Declare(
    miniaudio
    GIT_REPOSITORY https://github.com/mackron/miniaudio.git
    GIT_TAG        0.11.25
)

# we need to set these here so that miniaudio compilation doesn't compile the extra features
# When we do FetchContent_MakeAvailable, it will compile miniaudio with its default flags , so we do this so that it needn't do it
# But we need to define them in the header also so that they don't get compiled for our odai target

set(MINIAUDIO_NO_LIBVORBIS ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_LIBOPUS ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_EXTRA_NODES ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_DEVICEIO ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_RESOURCE_MANAGER ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_NODE_GRAPH ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_ENGINE ON CACHE BOOL "" FORCE)
set(MINIAUDIO_NO_GENERATION ON CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(miniaudio)
endif()

if(ODAI_ENABLE_STB_IMAGE)
    message(STATUS ">>> Enabling stb_image Image decoder")
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(stb)
endif()

# BUILD_SHARED_LIBS default for our project
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# -----------------------------------------------------------------
# 5. Core Library (The Main Target)
# -----------------------------------------------------------------
set(ODAI_CORE_SOURCES
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

add_library(odai SHARED ${ODAI_CORE_SOURCES})

# Core Headers & Definitions
target_include_directories(odai PUBLIC src/include)
target_include_directories(odai PRIVATE ${nlohmann_json_SOURCE_DIR})
target_include_directories(odai PRIVATE ${xxHash_SOURCE_DIR})

target_compile_definitions(odai PRIVATE BUILDING_ODAI_SHARED)
# Define XXH_INLINE_ALL to include all xxHash implementations in our build as header only lib, ensuring we get the performance benefits of inlining.
target_compile_definitions(odai PRIVATE XXH_INLINE_ALL)

# Core linking
target_link_libraries(odai PRIVATE nlohmann_json)

# -----------------------------------------------------------------
# 6. Optional Components Integration
# -----------------------------------------------------------------

if(ODAI_ENABLE_LLAMA_BACKEND)
    target_sources(odai PRIVATE src/impl/backendEngine/odai_llama_backend_engine.cpp)
    target_include_directories(odai PRIVATE ${llama_SOURCE_DIR})
    target_include_directories(odai PRIVATE ${llama_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(odai PRIVATE llama mtmd)
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_LLAMA_BACKEND)
endif()

if(ODAI_ENABLE_SQLITE_DB)
    target_sources(odai PRIVATE src/impl/db/odai_sqlite_db.cpp)
    target_include_directories(odai PRIVATE ${SQLiteCpp_SOURCE_DIR})
    target_link_libraries(odai PRIVATE sqlite3 sqlite-vec SQLiteCpp)
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_SQLITE_DB)
endif()

if(ODAI_ENABLE_MINIAUDIO)
    target_sources(odai PRIVATE src/impl/audioEngine/odai_miniaudio_decoder.cpp)
    target_include_directories(odai PRIVATE ${miniaudio_SOURCE_DIR})
    # Miniaudio is header only, no target linking needed
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_MINIAUDIO)

    target_compile_definitions(odai PRIVATE MA_NO_VORBIS)
    target_compile_definitions(odai PRIVATE MA_NO_OPUS)
    target_compile_definitions(odai PRIVATE MA_NO_DEVICE_IO)
    target_compile_definitions(odai PRIVATE MA_NO_RESOURCE_MANAGER)
    target_compile_definitions(odai PRIVATE MA_NO_NODE_GRAPH)
    target_compile_definitions(odai PRIVATE MA_NO_ENGINE)
    target_compile_definitions(odai PRIVATE MA_NO_GENERATION)

endif()

if(ODAI_ENABLE_STB_IMAGE)
    target_sources(odai PRIVATE src/impl/imageEngine/odai_stb_image_decoder.cpp)
    target_include_directories(odai PRIVATE ${stb_SOURCE_DIR})
    # stb is header only, no target linking needed
    target_compile_definitions(odai PRIVATE ODAI_ENABLE_STB_IMAGE)
endif()

# -----------------------------------------------------------------
# 7. Library Properties & Optimization
# -----------------------------------------------------------------

if(iporesult)
    message(STATUS "IPO/LTO is supported: enabling")
    # No need to set IPO for header only libraries like miniaudio
    # because they are included as source files from the source file which they are included in.
    # so they get optimized as part of the lib including source files.
    
    set_target_properties(odai PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(nlohmann_json PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

    if(ODAI_ENABLE_LLAMA_BACKEND)
        set_target_properties(llama PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
    
    if(ODAI_ENABLE_SQLITE_DB)
        set_target_properties(sqlite3 PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(sqlite-vec PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_target_properties(SQLiteCpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

else()
    message(WARNING "IPO/LTO not supported: ${output}")
endif()

# -----------------------------------------------------------------
# 8. Android JNI Support
# -----------------------------------------------------------------
if(ANDROID)
    set(ODAI_JNI_SOURCES
        src/impl/odai_jni.cpp
    )
    add_library(odaiAndroid SHARED ${ODAI_JNI_SOURCES})
    target_include_directories(odaiAndroid PUBLIC src/include)
    target_link_libraries(odaiAndroid PRIVATE odai)
endif()