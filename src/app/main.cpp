#include <GLFW/glfw3.h>
#include <engine/Engine.hpp>
#include <engine/Logger.hpp>
#include <engine/assert.hpp>

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

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        renderer.resizeExtent(width, height);
        renderer.drawFrame();
    }
    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
}
