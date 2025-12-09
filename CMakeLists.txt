cmake_minimum_required(VERSION 3.10)
project(Door-Automation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES
    src/*.cpp
    include/*.cpp
)

add_executable(Door_Automation ${SRC_FILES})

find_package(CURL REQUIRED)
target_link_libraries(Door_Automation PRIVATE ${CURL_LIBRARIES})

find_package(Threads REQUIRED)
target_link_libraries(Door_Automation PRIVATE Threads::Threads)

target_link_libraries(Door_Automation PRIVATE gpiodcxx)

target_include_directories(Door_Automation PRIVATE include)

set_target_properties(Door_Automation PROPERTIES OUTPUT_NAME "Door-Automation")
