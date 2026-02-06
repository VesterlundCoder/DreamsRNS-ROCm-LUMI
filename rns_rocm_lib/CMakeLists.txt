cmake_minimum_required(VERSION 3.21)
project(rocm_rns_matmul LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(RNS_ENABLE_GPU "Enable GPU (ROCm/HIP) support" ON)
option(RNS_ENABLE_TESTS "Enable unit tests" ON)
option(RNS_ENABLE_EXAMPLES "Enable example programs" ON)
option(RNS_BUILD_PYTHON "Build Python bindings" OFF)

# Find Boost for multiprecision (header-only)
find_package(Boost QUIET)
if(Boost_FOUND)
  message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}")
  add_definitions(-DRNS_HAS_BOOST)
else()
  message(STATUS "Boost not found - using internal bigint implementation")
endif()

# GPU support via HIP
if(RNS_ENABLE_GPU)
  # Try to find HIP
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/hip")
  find_package(hip QUIET)
  
  if(hip_FOUND)
    message(STATUS "Found HIP: ${hip_VERSION}")
    enable_language(HIP)
    set(RNS_HAS_GPU TRUE)
    add_definitions(-DRNS_HAS_GPU)
  else()
    message(WARNING "HIP not found - building CPU-only version")
    set(RNS_HAS_GPU FALSE)
    set(RNS_ENABLE_GPU OFF)
  endif()
else()
  set(RNS_HAS_GPU FALSE)
endif()

# Source files
set(RNS_SOURCES
  src/rns_api.cpp
  src/crt.cpp
  src/utils.cpp
  src/primes.cpp
  src/io.cpp
  src/rns_eval_cpu.cpp
  src/rns_walk_cpu.cpp
  src/topk_cpu.cpp
  src/c_api.cpp
)

if(RNS_HAS_GPU)
  list(APPEND RNS_SOURCES
    src/hip/rns_kernels.hip
    src/hip/rns_matmul_m4.hip
    src/hip/rns_matmul_m6.hip
    src/hip/rns_matmul_m8.hip
    src/hip/rns_matmul_m10.hip
    src/hip/modops.hip
    src/hip/rns_eval_bytecode.hip
    src/hip/rns_walk_fused.hip
    src/hip/topk.hip
  )
endif()

# Main library (renamed for Python to avoid conflict)
add_library(rns_rocm_lib ${RNS_SOURCES})

# Alias for backwards compatibility
add_library(rns_rocm ALIAS rns_rocm_lib)

target_include_directories(rns_rocm_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(Boost_FOUND)
  target_include_directories(rns_rocm_lib PUBLIC ${Boost_INCLUDE_DIRS})
endif()

if(RNS_HAS_GPU)
  # LUMI-G MI250X uses gfx90a exclusively.
  # Override with -DCMAKE_HIP_ARCHITECTURES=... if needed.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
  endif()
  set_target_properties(rns_rocm_lib PROPERTIES
    HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
  )
  target_link_libraries(rns_rocm_lib PUBLIC hip::device)
endif()

# Tests
if(RNS_ENABLE_TESTS)
  enable_testing()
  
  # CPU tests (always build)
  add_executable(test_rns_ops tests/test_rns_ops.cpp)
  target_link_libraries(test_rns_ops PRIVATE rns_rocm)
  add_test(NAME test_rns_ops COMMAND test_rns_ops)
  
  add_executable(test_crt tests/test_crt.cpp)
  target_link_libraries(test_crt PRIVATE rns_rocm)
  add_test(NAME test_crt COMMAND test_crt)
  
  add_executable(test_crt_matrix tests/test_crt_matrix.cpp)
  target_link_libraries(test_crt_matrix PRIVATE rns_rocm)
  add_test(NAME test_crt_matrix COMMAND test_crt_matrix)
  
  if(RNS_HAS_GPU)
    add_executable(test_gemm_mod tests/test_gemm_mod.cpp)
    target_link_libraries(test_gemm_mod PRIVATE rns_rocm)
    add_test(NAME test_gemm_mod COMMAND test_gemm_mod)
  endif()
  
  # New tests for v2 modules
  add_executable(test_primes tests/test_primes.cpp)
  target_link_libraries(test_primes PRIVATE rns_rocm)
  add_test(NAME test_primes COMMAND test_primes)
  
  add_executable(test_modops tests/test_modops.cpp)
  target_link_libraries(test_modops PRIVATE rns_rocm)
  add_test(NAME test_modops COMMAND test_modops)
  
  add_executable(test_eval_bytecode tests/test_eval_bytecode.cpp)
  target_link_libraries(test_eval_bytecode PRIVATE rns_rocm)
  add_test(NAME test_eval_bytecode COMMAND test_eval_bytecode)
  
  add_executable(test_walk tests/test_walk.cpp)
  target_link_libraries(test_walk PRIVATE rns_rocm)
  add_test(NAME test_walk COMMAND test_walk)
  
  add_executable(test_topk tests/test_topk.cpp)
  target_link_libraries(test_topk PRIVATE rns_rocm)
  add_test(NAME test_topk COMMAND test_topk)
  
  add_executable(test_montgomery tests/test_montgomery.cpp)
  target_link_libraries(test_montgomery PRIVATE rns_rocm)
  add_test(NAME test_montgomery COMMAND test_montgomery)
  
  add_executable(test_crt_approx tests/test_crt_approx.cpp)
  target_link_libraries(test_crt_approx PRIVATE rns_rocm)
  add_test(NAME test_crt_approx COMMAND test_crt_approx)
endif()

# Examples
if(RNS_ENABLE_EXAMPLES)
  if(RNS_HAS_GPU)
    add_executable(demo_gemm examples/demo_gemm.cpp)
    target_link_libraries(demo_gemm PRIVATE rns_rocm)
  endif()
  
  add_executable(demo_crt_matrix examples/demo_crt_matrix.cpp)
  target_link_libraries(demo_crt_matrix PRIVATE rns_rocm)
  
  add_executable(demo_cmf_pipeline examples/demo_cmf_pipeline.cpp)
  target_link_libraries(demo_cmf_pipeline PRIVATE rns_rocm)
endif()

# Python bindings (optional)
if(RNS_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Install
install(TARGETS rns_rocm_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/rns DESTINATION include)

# Summary
message(STATUS "")
message(STATUS "=== RNS-RoCM Configuration ===")
message(STATUS "  GPU support:    ${RNS_HAS_GPU}")
message(STATUS "  Tests:          ${RNS_ENABLE_TESTS}")
message(STATUS "  Examples:       ${RNS_ENABLE_EXAMPLES}")
message(STATUS "  Python:         ${RNS_BUILD_PYTHON}")
message(STATUS "  Boost bigint:   ${Boost_FOUND}")
message(STATUS "")
