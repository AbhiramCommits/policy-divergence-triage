cmake_minimum_required(VERSION 3.20)
project(pdt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PDT_BUILD_TESTS "Build the GoogleTest unit test suite" ON)
option(PDT_BUILD_PYTHON "Build the pybind11 extension module pdt_core" ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.13.6
)

FetchContent_MakeAvailable(nlohmann_json)

add_library(pdt_core_lib STATIC
  cpp/src/planner.cpp
  cpp/src/scenario.cpp
)
target_include_directories(pdt_core_lib PUBLIC cpp/include)
target_link_libraries(pdt_core_lib PUBLIC nlohmann_json::nlohmann_json)

if(PDT_BUILD_PYTHON)
  FetchContent_MakeAvailable(pybind11)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  pybind11_add_module(pdt_core cpp/bindings/pdt_core.cpp)
  target_link_libraries(pdt_core PRIVATE pdt_core_lib)
  install(TARGETS pdt_core LIBRARY DESTINATION .)
endif()

if(PDT_BUILD_TESTS)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(pdt_tests
    cpp/tests/test_planner.cpp
    cpp/tests/test_scenario_io.cpp
  )
  target_link_libraries(pdt_tests PRIVATE pdt_core_lib GTest::gtest_main)
  target_compile_definitions(pdt_tests PRIVATE
    PDT_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
  )
  include(GoogleTest)
  gtest_discover_tests(pdt_tests)
endif()
