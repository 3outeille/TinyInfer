FUNCTION(setup_eigen)
    execute_process(
            COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download
    )
    if (result)
        message(FATAL_ERROR "CMake step for eigen failed: ${result}")
    endif ()

    execute_process(
            COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/eigen-download
    )
    if (result)
        message(FATAL_ERROR "Build step for eigen failed: ${result}")
    endif ()
    list(APPEND ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/eigen-build)
    find_package(Eigen3 REQUIRED)
ENDFUNCTION()
