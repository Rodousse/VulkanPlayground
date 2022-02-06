set(ENGINE_SRC
  engine/Camera.cpp
  engine/DataIO.cpp
  engine/DebugMessenger.cpp
  engine/descriptor.cpp
  engine/Engine.cpp
  engine/Mesh.cpp
  engine/PerspectiveCamera.cpp
  engine/PhysicalDeviceProperties.cpp
  engine/utils.cpp)

set(ENGINE_HEADERS
  engine/assert.hpp
  engine/Camera.hpp
  engine/DataIO.hpp
  engine/DebugMessenger.hpp
  engine/descriptor.hpp
  engine/Engine.hpp
  engine/Logger.hpp
  engine/Mesh.hpp
  engine/PerspectiveCamera.hpp
  engine/PhysicalDeviceProperties.hpp
  engine/utils.hpp)

add_library(engine SHARED ${ENGINE_SRC} ${ENGINE_HEADERS})
target_link_libraries(engine PUBLIC imgui Vulkan::Vulkan Eigen3::Eigen Stbipp::Stbipp PRIVATE assimp::assimp)
target_include_directories(engine PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${GENERATED_DIR}>
  )

generate_export_header(engine
    EXPORT_MACRO_NAME ENGINE_API
    EXPORT_FILE_NAME ${GENERATED_DIR}/EngineSymbols.h
  )

set(ENGINE_STATIC_FLAG "ENGINE_STATIC_DEFINE")
if(BUILD_SHARED_LIBS)
  set(ENGINE_STATIC_FLAG"")
endif()

target_compile_definitions(engine PUBLIC ${ENGINE_STATIC_FLAG} _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE PRIVATE ENGINE_EXPORT)
target_compile_shaders(engine "${SHADER_SRC_DIR}/vertex.vert;${SHADER_SRC_DIR}/fragment.frag")
target_compile_definitions(engine PRIVATE -DMESH_PATH="${RESOURCE_DIR}/meshes/bunny.ply" -DSHADER_PATH="${SHADER_BINARY_DIR}")
