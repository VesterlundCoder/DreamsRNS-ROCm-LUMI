# Python bindings CMakeLists.txt
# This is included from the main CMakeLists.txt when RNS_BUILD_PYTHON=ON

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Try to fetch pybind11 if not found
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Python module
pybind11_add_module(rns_rocm
    rns_bindings.cpp
)

target_link_libraries(rns_rocm PRIVATE rns_rocm_lib)
target_include_directories(rns_rocm PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Set output name without 'lib' prefix
set_target_properties(rns_rocm PROPERTIES
    PREFIX ""
    OUTPUT_NAME "rns_rocm"
)

# Install to Python package directory
install(TARGETS rns_rocm
    LIBRARY DESTINATION ${Python_SITEARCH}
)
