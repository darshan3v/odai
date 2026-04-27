include(GoogleTest)

function(configure_audio_decoder_test target source labels)
    add_executable(${target}
        ${source}
    )

    target_link_libraries(${target}
        PRIVATE
            odai
            GTest::gtest_main
    )

    target_include_directories(${target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    add_dependencies(${target} odai_test_data)

    target_compile_definitions(${target}
        PRIVATE
            TEST_DATA_DIR="${TEST_DATA_OUTPUT_DIR}"
    )

    gtest_discover_tests(${target}
        PROPERTIES
            LABELS "${labels}"
    )
endfunction()

function(configure_miniaudio_decoder_test target source labels)
    configure_audio_decoder_test(${target} ${source} "${labels}")

    target_compile_definitions(${target}
        PRIVATE
            ODAI_ENABLE_MINIAUDIO
    )
endfunction()

function(configure_miniaudio_decoder_tests implementation_target implementation_source implementation_label)
    configure_miniaudio_decoder_test(odai_audio_decoder_contract_tests odai_audio_decoder_contract_test.cpp
                                     "audio\\;integration\\;contract\\;${implementation_label}")
    configure_miniaudio_decoder_test(${implementation_target} ${implementation_source}
                                     "audio\\;integration\\;${implementation_label}")
endfunction()

if(ODAI_ENABLE_MINIAUDIO)
    configure_miniaudio_decoder_tests(odai_miniaudio_decoder_tests odai_miniaudio_decoder_test.cpp miniaudio)
endif()
