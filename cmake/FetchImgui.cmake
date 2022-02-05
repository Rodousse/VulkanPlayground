function(fetch_imgui_target)
    FetchContent_Declare(
      imguirep
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG        "v1.86"
      GIT_SHALLOW    ON
    )

    FetchContent_GetProperties(imguirep)
    if(NOT imguirep_POPULATED)
      FetchContent_Populate(imguirep)
      set(imgui_SRC
        ${imguirep_SOURCE_DIR}/backends/imgui_impl_glfw.cpp 
        ${imguirep_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp 
        ${imguirep_SOURCE_DIR}/imgui.cpp 
        ${imguirep_SOURCE_DIR}/imgui_demo.cpp
        ${imguirep_SOURCE_DIR}/imgui_draw.cpp 
        ${imguirep_SOURCE_DIR}/imgui_tables.cpp 
        ${imguirep_SOURCE_DIR}/imgui_widgets.cpp)
    endif()
    add_library(imgui SHARED ${imgui_SRC})
    target_include_directories(imgui PUBLIC ${imguirep_SOURCE_DIR} ${imguirep_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
    target_compile_definitions(imgui PUBLIC "-DGLFW_INCLUDE_NONE -DGLFW_INCLUDE_VULKAN -DImTextureID=ImU64")
    message(STATUS ${imguirep_SOURCE_DIR})
    message(STATUS  ${imgui_file_dialog_SOURCE_DIR})


endfunction()
