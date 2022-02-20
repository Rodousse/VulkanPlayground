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
    } m_swapchainData;

    struct MeshData
    {
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
    } m_meshData;

    const std::vector<const char*> DEVICE_EXTENSIONS = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    const int MAX_FRAMES_IN_FLIGHT = 2;
    VkInstance m_instance;
    std::vector<const char*> m_requiredExtensions;
    VkPhysicalDeviceFeatures m_requiredDeviceFeatures{};

    SwapchainSupportDetails m_swapchainDetails;

    QueueFamilyIndices m_indices;

    VkPhysicalDeviceProperties m_deviceProperties;
    VkPhysicalDeviceMemoryProperties m_memoryProperties;
    VkPhysicalDevice m_physicalDevice;

    VkDevice m_logicalDevice;

    VkDebugUtilsMessengerEXT m_debugMessenger;

    VkSurfaceKHR m_surface;

    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    VkQueue m_transfertQueue;

    VkExtent2D m_windowExtent;

    VkRenderPass m_renderPass;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorPool m_imguiDescriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;
    VkViewport m_viewport;

    VkCommandPool m_commandPool;
    VkCommandPool m_commandPoolTransfert;
    std::vector<VkCommandBuffer> m_commandBufferSecondary;
    std::vector<VkCommandBuffer> m_commandBuffers;

    // used to synchronise the image to show
    std::vector<VkSemaphore> m_imageAvailableSemaphore; // An image is ready to render
    std::vector<VkSemaphore> m_renderFinishedSemaphore; // An image is rendered and wait to be presented
    std::vector<VkFence> m_inFlightFences;
    size_t m_currentFrame = 0;

    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformBuffersMemory;

    VkImage m_depthImage;
    VkImageView m_depthImageView;
    VkDeviceMemory m_depthImageMemory;

    VkImage m_colorImage;
    VkDeviceMemory m_colorMemory;
    VkImageView m_colorImageView;

    VkSampleCountFlagBits m_msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    bool m_framebufferResize = false;
    bool m_isCleaned = true;

    VkResult areInstanceExtensionsCompatible(const char** extensions, uint32_t extensionsCount);
    VkFormat findDepthFormat();
    VkSampleCountFlagBits getMaxUsableSampleCount();

    /******************************************* APPLICATION VARIABLE
     * ******************************************************/

    //    MaterialTexture lenaTexture_;
    std::shared_ptr<Camera> m_camera;
    Mesh m_mesh;
    ApplicationStateChange m_applicationChanges;

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
    void createSecondaryCommandBuffers();
    void createPrimaryCommandBuffer();
    void updatePrimaryCommandBuffers();
    void createDepthRessources();
    void createColorRessources();
    void initImgui();

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

  public:
    Engine();
    Engine(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine& operator=(Engine&&) = delete;

    /******************************************* APPLICATION FUNCTIONS
     * ******************************************************/

    void createInstance();
    VkInstance getInstance();
    void addRequiredExtensions(const char** extensions, uint32_t extensionCount);
    void setSurface(const VkSurfaceKHR& surface);
    void initVulkan();

    void resizeExtent(int width, int height);

    void setCamera(std::shared_ptr<Camera> camera);
    void setModel(const Mesh& model);

    void drawFrame();
    void cleanup();

    /***********************************************************************************************************************/

    ~Engine();
};

} // namespace engine
