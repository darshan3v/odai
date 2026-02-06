cmake_minimum_required(VERSION 3.9...3.10)

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
set(CMAKE_CXX_STANDARD 20) # llama.cpp usually requires C++17 or newer
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Critical for linking static libs into a shared .so

# if("${ANDROID_ABI}" STREQUAL "arm64-v8a")
#     add_compile_options(-march=armv8.2-a+dotprod)
# endif()


# -----------------------------------------------------------------
# 2. Dependencies (Static Linking)
# -----------------------------------------------------------------

include(FetchContent)

# --- Setup Llama.cpp ---
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

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

set_target_properties(llama PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
unset(BUILD_SHARED_LIBS)

# --- Setup SQLiteCPP ---
# This downloads SQLiteCpp AND the sqlite3 C library
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
unset(BUILD_SHARED_LIBS)

# --- Setup SQLite-Vec ---
# Assuming sqlite-vec has a simple C source file or CMake.
# If it has no CMake, we build it manually here:    
add_library(sqlite-vec STATIC sqlite-vec-0.1.6-amalgamation/sqlite-vec.c)
target_include_directories(sqlite-vec PRIVATE sqlite-vec-0.1.6-amalgamation)

# sqlite-vec needs to see sqlite3 headers
    target_link_libraries(sqlite-vec PRIVATE sqlite3)

# --- Setup nlohmann/json ---
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)
unset(BUILD_SHARED_LIBS)

# --- Setup xxHash ---
FetchContent_Declare(
    xxHash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.2
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xxHash)
unset(BUILD_SHARED_LIBS)

# -----------------------------------------------------------------
# 3. Your Library (The Main Target)
# -----------------------------------------------------------------

# Define your source files
set(ODAI_SOURCES
    src/impl/odai_logger.cpp
    src/impl/odai_public.cpp
    src/impl/odai_sdk.cpp

    src/impl/types/odai_type_conversions.cpp

    src/impl/utils/odai_helpers.cpp
    src/impl/utils/string_utils.cpp

    src/impl/backendEngine/odai_llama_backend_engine.cpp

    src/impl/db/odai_sqlite_db.cpp

    src/impl/ragEngine/odai_rag_engine.cpp
)

# Create a SHARED library.
# Android needs a .so (Shared) file to load via System.loadLibrary().
# We will link the static dependencies INTO this shared library.
add_library(odai SHARED ${ODAI_SOURCES})

# Include your headers
target_include_directories(odai PUBLIC src/include)

target_compile_definitions(odai PRIVATE BUILDING_ODAI_SHARED)

if (CMAKE_BUILD_TYPE STREQUAL "Release")

    target_compile_options(odai PRIVATE "-fvisibility=hidden")
    target_compile_options(odai PRIVATE "-fvisibility-inlines-hidden")

    # Exclude static lib symbols from export
    set_target_properties(odai PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
endif()

if(iporesult)
    message(STATUS "IPO/LTO is supported: enabling")
    # Apply to your main library
    set_target_properties(odai PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    
    # Apply to dependencies (Critical for static linking!)
    set_target_properties(llama PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(sqlite3 PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(sqlite-vec PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(SQLiteCpp PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_properties(nlohmann_json PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO/LTO not supported: ${output}")
endif()

# -----------------------------------------------------------------
# 4. Linking Everything Together
# -----------------------------------------------------------------

# Link the static libraries into your shared library.
# The linker will copy the machine code from the .a files into your .so
target_link_libraries(odai
    PRIVATE
    llama       # From llama.cpp directory
    sqlite3
    sqlite-vec
    SQLiteCpp
    nlohmann_json
    xxhash
)

if(ANDROID)

set(ODAI_JNI_SOURCES
    src/impl/odai_jni.cpp
)

add_library(odaiAndroid SHARED ${ODAI_JNI_SOURCES})
# set_target_properties(odaiAndroid PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    target_include_directories(odaiAndroid PUBLIC src/include)

    target_link_libraries(
            odaiAndroid
            PRIVATE
            odai
    )
endif()