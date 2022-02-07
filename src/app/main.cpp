#include <GLFW/glfw3.h>
#include <engine/Engine.hpp>
#include <engine/Logger.hpp>
#include <engine/assert.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

int main()
{
    if(!glfwInit())
    {
        LOG_ERROR("Could not init glfw");
        return 1;
    }

    if(!glfwVulkanSupported())
    {
        LOG_ERROR("GLFW does not support vulkan");
        return 1;
    }

    engine::Engine renderer;
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    auto* window = glfwCreateWindow(1280, 720, "Vulkan Playground", nullptr, nullptr);
    uint32_t extensionCount{0};
    const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);

    renderer.addRequiredExtensions(extensions, extensionCount);

    renderer.createInstance();

    VkSurfaceKHR surface;
    VK_CALL(glfwCreateWindowSurface(renderer.getInstance(), window, nullptr, &surface));
    renderer.setSurface(surface);
    renderer.initVulkan();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    static bool show_demo_window{true};

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        renderer.resizeExtent(width, height);

        ImGui::ShowDemoWindow(&show_demo_window);

        ImGui::Render();

        renderer.drawFrame();
        // Start the Dear ImGui frame
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
}
