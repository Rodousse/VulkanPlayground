
# Credits to @evilactually on github
# https://gist.github.com/evilactually/a0d191701cb48f157b05be7f74d79396

set(GLSL_VALIDATOR_PATH "$ENV{VULKAN_SDK}/bin")
if(WIN32)
  if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    set(GLSL_VALIDATOR_PATH "$ENV{VULKAN_SDK}/Bin")
  else()
    set(GLSL_VALIDATOR_PATH "$ENV{VULKAN_SDK}/Bin32")
  endif()
endif()

set(GLSL_VALIDATOR "${GLSL_VALIDATOR_PATH}/glslangValidator${CMAKE_EXECUTABLE_SUFFIX}")


function(target_compile_shaders TARGET_ SHADER_PATHS)

  foreach(shader ${SHADER_PATHS})
    get_filename_component(shaderFileName ${shader} NAME_WE)
    set(binary_path ${SHADER_BINARY_DIR}/${shaderFileName}.spv)
    add_custom_command(
        OUTPUT ${binary_path}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_BINARY_DIR}"
        COMMAND ${GLSL_VALIDATOR} -V ${shader} -o ${binary_path}
        DEPENDS ${shader})
    list(APPEND SPIRV_BINARY_FILES ${binary_path})
  endforeach(shader)

  add_custom_target(
    Shaders_${TARGET_} 
    DEPENDS ${SPIRV_BINARY_FILES}
    )

  add_dependencies(${TARGET_} Shaders_${TARGET_})

  #  add_custom_command(TARGET ${TARGET_} POST_BUILD
  #      COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:YourMainTarget>/shaders/"
  #      COMMAND ${CMAKE_COMMAND} -E copy_directory
  #          "${PROJECT_BINARY_DIR}/shaders"
  #          "$<TARGET_FILE_DIR:YourMainTarget>/shaders"
  #          )

endfunction()
