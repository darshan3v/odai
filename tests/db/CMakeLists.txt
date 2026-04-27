include(GoogleTest)

function(configure_db_test target source labels)
    add_executable(${target} ${source})

    target_link_libraries(${target} PRIVATE odai GTest::gtest_main)

    target_include_directories(${target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/.."
    )

    gtest_discover_tests(${target}
        PROPERTIES
            LABELS "${labels}"
    )
endfunction()

function(configure_sqlite_db_test target source labels)
    configure_db_test(${target} ${source} "${labels}")

    target_compile_definitions(${target} PRIVATE ODAI_ENABLE_SQLITE_DB)
    target_include_directories(${target} PRIVATE ${SQLiteCpp_SOURCE_DIR}/include)
endfunction()

function(configure_sqlite_db_tests implementation_target implementation_source implementation_label)
    configure_sqlite_db_test(odai_sqlite_db_contract_tests odai_sqlite_db_contract_test.cpp
                             "db\\;integration\\;contract\\;${implementation_label}")
    configure_sqlite_db_test(${implementation_target} ${implementation_source}
                             "db\\;integration\\;${implementation_label}")
endfunction()

if(ODAI_ENABLE_SQLITE_DB)
    configure_sqlite_db_tests(odai_sqlite_db_tests odai_sqlite_db_test.cpp sqlite)
endif()
