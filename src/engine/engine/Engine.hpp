#pragma once
#include "engine/Camera.hpp"
#include "engine/CommonTypes.hpp"
#include "engine/DebugMessenger.hpp"
#include "engine/Mesh.hpp"
#include "engine/PhysicalDeviceProperties.hpp"

#include <EngineSymbols.h>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef WIN32_
#define VK_USE_PLATFORM_WIN32_KHR
#elif UNIX_
#define VK_USE_PLATFORM_XCB_KHR
#endif

namespace engine
{
#ifdef NDEBUG
const bool ENABLE_VALIDATION_LAYERS = false;
#else
const bool ENABLE_VALIDATION_LAYERS = true;
#endif

class Engine
{
  private:
    /******************************************* CORE VARIABLE ******************************************************/
    struct ApplicationStateChange
    {
        bool materialModified = false;
        bool modelModified = false;
    };

    struct UniformBufferObject
    {
        Matrix4 model;
        Matrix4 view;
        Matrix4 projection;
        Vector3 lightPos;
    };

    struct Swapchain
    {
        VkSwapchainKHR swapchain;
        std::vector<VkImage> images;
        std::vector<VkImageView> imageViews;
        std::vector<VkFramebuffer> framebuffers;

        VkSurfaceFormatKHR format;
        VkExtent2D extent;
        VkPresentModeKHR presentMode;
    } swapchainData_;

    struct MeshData
    {
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
    } meshData_;

    const std::vector<const char*> DEVICE_EXTENSIONS = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    const int MAX_FRAMES_IN_FLIGHT = 2;
    VkInstance instance_;
    std::vector<const char*> requiredExtensions_;
    VkPhysicalDeviceFeatures requiredDeviceFeatures_;

    SwapchainSupportDetails swapchainDetails_;

    QueueFamilyIndices indices_;

    VkPhysicalDeviceProperties deviceProperties_;
    VkPhysicalDeviceMemoryProperties memoryProperties_;
    VkPhysicalDevice physicalDevice_;

    VkDevice logicalDevice_;

    VkDebugUtilsMessengerEXT debugMessenger_;

    VkSurfaceKHR surface_;

    VkQueue graphicsQueue_;
    VkQueue presentQueue_;
    VkQueue transfertQueue_;

    VkExtent2D windowExtent_;
    SwapchainSupportDetails swapchainSupportDetails_;

    VkRenderPass renderPass_;
    VkDescriptorSetLayout descriptorSetLayout_;
    VkDescriptorPool descriptorPool_;
    std::vector<VkDescriptorSet> descriptorSets_;
    VkPipelineLayout pipelineLayout_;
    VkPipeline graphicsPipeline_;
    VkViewport viewport_;

    VkCommandPool commandPool_;
    VkCommandPool commandPoolTransfert_;
    std::vector<VkCommandBuffer> commandBuffers_;

    // used to synchronise the image to show
    std::vector<VkSemaphore> imageAvailableSemaphore_; // An image is ready to render
    std::vector<VkSemaphore> renderFinishedSemaphore_; // An image is rendered and wait to be presented
    std::vector<VkFence> inFlightFences_;
    size_t currentFrame_ = 0;

    VkBuffer vertexBuffer_;
    VkDeviceMemory vertexBufferMemory_;
    VkBuffer vertexIndexBuffer_;
    VkDeviceMemory vertexIndexBufferMemory_;
    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;

    VkImage depthImage_;
    VkImageView depthImageView_;
    VkDeviceMemory depthImageMemory_;

    VkImage colorImage_;
    VkDeviceMemory colorMemory_;
    VkImageView colorImageView_;

    VkSampleCountFlagBits msaaSamples_ = VK_SAMPLE_COUNT_1_BIT;

    bool framebufferResize = false;
    bool isCleaned_ = true;

    VkResult areInstanceExtensionsCompatible(const char** extensions, uint32_t extensionsCount);
    VkFormat findDepthFormat();
    VkSampleCountFlagBits getMaxUsableSampleCount();

    /******************************************* APPLICATION VARIABLE
     * ******************************************************/

    MaterialTexture lenaTexture_;
    std::shared_ptr<Camera> camera_;
    Mesh mesh_;
    ApplicationStateChange applicationChanges_;

    /***********************************************************************************************************************/

    // Physical devices and Queues compatibility

    void pickPhysicalDevice();
    void createLogicalDevice();

    // Swapchains, graphics pipeline

    void createSwapChain();
    void recreateSwapChain();
    void cleanUpSwapChain();
    void createRenderPass();
    void createGraphicsPipeline();
    void createFramebuffers(VkRenderPass renderPass, const std::vector<VkImageView>& attachements);
    void createCommandPool();
    void createCommandBuffers();
    void createDepthRessources();
    void createColorRessources();

    void recreateCommandBuffer();

    [[nodiscard]] VkCommandPool getCommandPoolTransfer() const;
    [[nodiscard]] VkCommandPool getCommandPool() const;
    [[nodiscard]] VkQueue getGraphicsQueue() const;
    [[nodiscard]] VkQueue getTransfertQueue() const;
    // Shader Loading and Creation

    static std::vector<char> readFile(const std::string& fileName); // Read the content of the spv file
    VkShaderModule createShaderModule(const std::vector<char>& shaderCode);

    // Buffer Management

    void createVertexBuffer();
    void createVertexIndexBuffer();
    void createUniformBuffer();
    void updateUniformBuffer(uint32_t imageIndex);

    // DescriptorSetLayout and Uniform buffer

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();

    // Synchronisation
    void createSyncObjects();
    void checkApplicationState();

    // Swapchain
    void chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    void chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes);
    void chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

    void cleanup();

  public:
    Engine();
    Engine(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine& operator=(Engine&&) = delete;

    /******************************************* APPLICATION FUNCTIONS
     * ******************************************************/

    void createInstance();
    void addRequiredExtensions(const char** extensions, uint32_t extensionCount);
    void setSurface(const VkSurfaceKHR& surface);
    void initVulkan();

    void resizeExtent(int width, int height);

    void setCamera(std::shared_ptr<Camera> camera);
    void setModel(const Mesh& model);

    void drawFrame();

    /***********************************************************************************************************************/

    ~Engine();
};

} // namespace engine
