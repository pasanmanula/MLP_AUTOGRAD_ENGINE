# Add test executable
add_executable(CoreTest test_all.cpp test_mlp_engine/test_core.cpp)

# Link Google Test libraries to the test executable
target_link_libraries(CoreTest gtest gtest_main "${TORCH_LIBRARIES}")

# Include directories for the test target
target_include_directories(CoreTest PUBLIC ../include pytorch_libs)

# Add a test to CTest
add_test(NAME CoreTest COMMAND CoreTest)

# Note -> "cmake -DBUILD_TESTING=ON .."" to build test files :D