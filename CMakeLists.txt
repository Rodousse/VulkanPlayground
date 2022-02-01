cmake_minimum_required(VERSION 3.12)

project(VulkanPlayground LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Stbipp REQUIRED)
find_package(assimp REQUIRED)
find_package(glfw3 REQUIRED)

include(FetchContent)
include(GenerateExportHeader)
include(cmake/FetchImgui.cmake)

# Path to use for cmake generated files
set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
message(STATUS "Generated files goes in : ${GENERATED_DIR}")
file(MAKE_DIRECTORY ${GENERATED_DIR})

fetch_imgui_target()

add_subdirectory(src/engine)
add_subdirectory(src/app)
