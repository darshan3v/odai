include(GoogleTest)

function(configure_image_decoder_test target source labels)
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

function(configure_stb_image_decoder_test target source labels)
    configure_image_decoder_test(${target} ${source} "${labels}")

    target_compile_definitions(${target}
        PRIVATE
            ODAI_ENABLE_STB_IMAGE
    )
endfunction()

function(configure_stb_image_decoder_tests implementation_target implementation_source implementation_label)
    configure_stb_image_decoder_test(odai_image_decoder_contract_tests odai_image_decoder_contract_test.cpp
                                     "image\\;integration\\;contract\\;${implementation_label}")
    configure_stb_image_decoder_test(${implementation_target} ${implementation_source}
                                     "image\\;integration\\;${implementation_label}")
endfunction()

if(ODAI_ENABLE_STB_IMAGE)
    configure_stb_image_decoder_tests(odai_stb_image_decoder_tests odai_stb_image_decoder_test.cpp stb)
endif()
