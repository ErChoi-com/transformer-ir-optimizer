cmake_minimum_required(VERSION 3.20)

project(TransformerIROptimizer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_testing()

option(LLM_PASS_BUILD_BENCHMARKS "Build standalone transformer kernel benchmarks" ON)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(LLMInferencePass MODULE
  src/LLMInferencePass.cpp
)

target_include_directories(LLMInferencePass
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(LLMInferencePass PROPERTIES
  PREFIX ""
  OUTPUT_NAME "LLMInferencePass"
)

llvm_map_components_to_libnames(LLVM_LIBS
  Analysis
  Core
  IRReader
  Passes
  ScalarOpts
  Support
  TransformUtils
)

target_link_libraries(LLMInferencePass PRIVATE ${LLVM_LIBS})

find_program(LLVM_LIT_EXECUTABLE NAMES llvm-lit lit)
find_program(LLVM_OPT_EXECUTABLE NAMES opt)
find_program(LLVM_FILECHECK_EXECUTABLE NAMES FileCheck)

if(LLVM_LIT_EXECUTABLE AND LLVM_OPT_EXECUTABLE AND LLVM_FILECHECK_EXECUTABLE)
  add_test(
    NAME llm-pass-lit
    COMMAND ${LLVM_LIT_EXECUTABLE}
            -Dopt=${LLVM_OPT_EXECUTABLE}
            -DFileCheck=${LLVM_FILECHECK_EXECUTABLE}
            -Dshlibext=$<IF:$<PLATFORM_ID:Windows>,.dll,$<IF:$<PLATFORM_ID:Darwin>,.dylib,.so>>
            -Dshlibdir=$<TARGET_FILE_DIR:LLMInferencePass>
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/FileCheck
  )
endif()

if(LLM_PASS_BUILD_BENCHMARKS)
  add_executable(transformer_kernels
    benchmarks/driver.cpp
    benchmarks/transformer_kernels.cpp
  )

  target_include_directories(transformer_kernels
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/include
  )

  if(MSVC)
    target_compile_options(transformer_kernels PRIVATE /O2 /fp:fast)
  else()
    target_compile_options(transformer_kernels PRIVATE -O3 -ffast-math)
  endif()
endif()