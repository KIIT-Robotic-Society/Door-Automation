# minimum CMake version required for this project
cmake_minimum_required(VERSION 3.10)

project(Door-Automation)

# C++ Standard Configuration
# enforce C++17 for STL features, std::thread, filesystem, etc.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# automatically gathers all .cpp files from src/ and include/
# yhis allows adding new source files without modifying CMake.
file(GLOB SRC_FILES
    src/*.cpp
    include/*.cpp
)

# create Executable Target
add_executable(Door_Automation ${SRC_FILES})


# link libcurl (HTTP requests, APIs)
find_package(CURL REQUIRED)
target_link_libraries(Door_Automation PRIVATE ${CURL_LIBRARIES})

# link pthread / std::thread
find_package(Threads REQUIRED)
target_link_libraries(Door_Automation PRIVATE Threads::Threads)

# link libgpiodcxx (GPIO control for Linux)
target_link_libraries(Door_Automation PRIVATE gpiodcxx)

# include Project Header Files
target_include_directories(Door_Automation PRIVATE include)

# rename output binary to "Door-Automation"
set_target_properties(Door_Automation PROPERTIES OUTPUT_NAME "Door-Automation")
