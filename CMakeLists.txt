cmake_minimum_required(VERSION 3.31)
project(ComputationalPractice)

set(CMAKE_CXX_STANDARD 20)

add_executable(ComputationalPractice main.cpp
        VectorMath.h
        MatrixMath.h
        )

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

enable_testing()

add_executable(MyTests test.cpp
        VectorMath.h
        MatrixMath.h
)

target_link_libraries(MyTests PRIVATE gtest_main)

gtest_discover_tests(MyTests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        # ARGS --gtest_output=xml:report.xml
        # DISCOVERY_TIMEOUT 120
)

add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS MyTests
)

add_dependencies(ComputationalPractice run_tests)