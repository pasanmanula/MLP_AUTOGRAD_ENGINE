cmake_minimum_required(VERSION 3.5)  # CMake version check
project(mlp_autograd)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-missing-field-initializers -O3)
endif()

# Enable testing
enable_testing()

# Add Google Test subdirectory
add_subdirectory(googletest)

# Set the path to the PyTorch libraries
set(CMAKE_PREFIX_PATH pytorch_libs/libtorch)

# Find the Torch package
find_package(Torch REQUIRED)

# Set the C++ flags for Torch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the main executable target with source files
add_executable(${PROJECT_NAME}
  src/main.cpp)

# Include directories for the main project
target_include_directories(${PROJECT_NAME} PUBLIC include pytorch_libs)

# Link the Torch libraries to the main executable
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")

# Add the test directory if building tests --> instead of cmake .. -> use : cmake -DBUILD_TESTING=ON .. then make
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
