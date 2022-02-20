#include "engine/Engine.hpp"

#include "DataIO.hpp"
#include "engine/CommonTypes.hpp"
#include "engine/DebugMessenger.hpp"
#include "engine/Logger.hpp"
#include "engine/PhysicalDeviceProperties.hpp"
#include "engine/assert.hpp"
#include "engine/descriptor.hpp"
#include "engine/utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <imgui_impl_vulkan.h>
#include <set>
#include <vulkan/vulkan_core.h>

namespace
{
bool isDeviceSuitable(VkPhysicalDevice device,
                      VkSurfaceKHR surface,
                      const std::vector<const char*>& extensions,
                      const VkPhysicalDeviceFeatures& requiredFeatures);

VkPhysicalDevice getBestPhysicalDevice(VkInstance instance,
                                       VkSurfaceKHR surface,
                                       const std::vector<const char*>& extensions,
                                       const VkPhysicalDeviceFeatures& requiredFeatures);

} // namespace
namespace engine
{
Engine::Engine()
{
    m_requiredDeviceFeatures.sampleRateShading = VK_TRUE;
    auto scene = IO::loadScene(MESH_PATH);
    if(!scene)
    {
        THROW(std::runtime_error("Could not load the mesh"));
    }
    m_mesh = scene->meshes[0];
    m_camera = std::move(scene->cameras.front());
    if constexpr(ENABLE_VALIDATION_LAYERS)
    {
        m_requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
}

void Engine::drawFrame()
{
    checkApplicationState();

    uint32_t imageIndex;

    vkWaitForFences(
      m_logicalDevice, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame]);
    // Timeout in nanoseconds = numeric_limits... using the max disable the timeout
    VkResult result = vkAcquireNextImageKHR(m_logicalDevice,
                                            m_swapchainData.swapchain,
                                            std::numeric_limits<uint64_t>::max(),
                                            m_imageAvailableSemaphore[m_currentFrame],
                                            VK_NULL_HANDLE,
                                            &imageIndex);

    if(result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapChain();
        return;
    }
    else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(imageIndex);
    updatePrimaryCommandBuffers();

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &m_imageAvailableSemaphore[m_currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_renderFinishedSemaphore[m_currentFrame];

    VkPipelineStageFlags waitStageFlags[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.pWaitDstStageMask = waitStageFlags; // We wait the semaphore with to have the entry with the same index
    // Here we wait the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT of the m_imageAvailableSemaphore

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

    vkResetFences(m_logicalDevice, 1, &m_inFlightFences[m_currentFrame]);

    VK_CALL(vkQueueSubmit(m_presentQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]));

    VkResult presentatioResult;
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchainData.swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &m_renderFinishedSemaphore[m_currentFrame];
    presentInfo.pResults = &presentatioResult; // Array of results for each swap chain images

    result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResize)
    {
        m_framebufferResize = false;
        recreateSwapChain();
    }
    else if(result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    vkQueueWaitIdle(m_presentQueue);

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

Engine::~Engine()
{
    cleanup();
}

void Engine::addRequiredExtensions(const char** extensions, uint32_t extensionCount)
{
    m_requiredExtensions.insert(m_requiredExtensions.end(), extensions, extensions + extensionCount);
}

void Engine::initVulkan()
{
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthRessources();
    createColorRessources();
    createFramebuffers(m_renderPass, {m_colorImageView, m_depthImageView});
    // m_lenaTexture.create();
    createVertexBuffer();
    createVertexIndexBuffer();
    createUniformBuffer();
    createDescriptorPool();
    createDescriptorSets();
    createPrimaryCommandBuffer();
    createSecondaryCommandBuffers();
    createSyncObjects();
    initImgui();

    LOG_INFO("Vulkan Initialisation Finished");
}

void Engine::setSurface(const VkSurfaceKHR& surface)
{
    m_surface = surface;
}

VkQueue Engine::getTransfertQueue() const
{
    if(m_indices.transferAvailable())
    {
        return m_transfertQueue;
    }
    return getGraphicsQueue();
}

VkQueue Engine::getGraphicsQueue() const
{
    return m_graphicsQueue;
}

VkCommandPool Engine::getCommandPoolTransfer() const
{
    if(m_indices.transferAvailable())
    {
        return m_commandPoolTransfert;
    }
    return getCommandPool();
}

VkCommandPool Engine::getCommandPool() const
{
    return m_commandPool;
}

void Engine::createInstance()
{
    m_isCleaned = false;

    // check if the validation layers are needed and if they are available
    if(ENABLE_VALIDATION_LAYERS && !debug::checkValidationLayerSupport())
    {
        THROW(std::runtime_error("Validation layer requested, but not available !"));
    }

    VK_CALL(
      areInstanceExtensionsCompatible(m_requiredExtensions.data(), static_cast<uint32_t>(m_requiredExtensions.size())));

    LOG_INFO("Vulkan Instance Creation...");
    // Applications infos about the version, the engine version, the vulkan version used
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Arverne Viewer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "SuperViewerArverne";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // Informations relative to the appInfo and the window system used
    // And need the extensions to use from the window system (dependent of the library used)
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Validation layer precisions passed to the instance
    if constexpr(ENABLE_VALIDATION_LAYERS)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(debug::VALIDATION_LAYERS.size());
        createInfo.ppEnabledLayerNames = debug::VALIDATION_LAYERS.data();
    }
    else
    {
        createInfo.enabledLayerCount = 0;
    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(m_requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = m_requiredExtensions.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);

    VK_CALL(result);

    if constexpr(ENABLE_VALIDATION_LAYERS)
    {
        debug::createDebugMessenger(m_instance, m_debugMessenger, debug::defaultDebugCallback);
    }

    LOG_INFO("Vulkan Instance Created");
}

VkInstance Engine::getInstance()
{
    return m_instance;
}

void Engine::resizeExtent(int width, int height)
{
    if(m_windowExtent.width == width && m_windowExtent.height == height)
    {
        return;
    }
    m_framebufferResize = true;
    m_windowExtent.width = width;
    m_windowExtent.height = height;
    m_camera->setViewportDimensions(width, height);
    m_swapchainDetails = querySwapChainSupport(m_physicalDevice, m_surface); // Used for querySwapChainSupport
}

void Engine::setCamera(std::shared_ptr<Camera> camera)
{
    m_camera = std::move(camera);
}

VkResult Engine::areInstanceExtensionsCompatible(const char** extensions, uint32_t extensionsCount)
{
    // How many extensions are available with vulkan
    uint32_t vkExtensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &vkExtensionCount, nullptr);

    // Store them
    std::vector<VkExtensionProperties> vkExtensions(vkExtensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &vkExtensionCount, vkExtensions.data());

    LOG_INFO("Number of vulkan extensions available " << vkExtensionCount);

    for(const auto& extension: vkExtensions)
    {
        LOG_INFO("\t" << extension.extensionName);
    }

    // If an extension passed in the parameter is not in the list of the vulkan available extensions, we can't create
    // the instance
    for(uint8_t i = 0; i < extensionsCount; i++)
    {
        if(std::find_if(vkExtensions.begin(),
                        vkExtensions.end(),
                        [extensions, i](const VkExtensionProperties& prop)
                        { return strcmp(prop.extensionName, extensions[i]) == 0; }) == vkExtensions.end())
        {
            std::cerr << "Extension : " << extensions[i] << " not supported" << std::endl;
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    return VK_SUCCESS;
}

VkFormat Engine::findDepthFormat()
{
    return findSupportedTilingFormat(m_physicalDevice,
                                     {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                     VK_IMAGE_TILING_OPTIMAL,
                                     VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void Engine::pickPhysicalDevice()
{
    LOG_INFO("Picking a physical device");

    m_physicalDevice = getBestPhysicalDevice(m_instance, m_surface, DEVICE_EXTENSIONS, m_requiredDeviceFeatures);

    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_memoryProperties);
    m_msaaSamples = getMaxUsableSampleCount();
    LOG_INFO(m_deviceProperties.deviceName << " : I chose you !!!!");
}

void Engine::createLogicalDevice()
{
    LOG_INFO("Creating a logical device...");
    m_indices = findQueueFamilies(m_physicalDevice, m_surface);
    float queuePriority = 1.0f;

    // Specifying the queues to be created
    std::vector<VkDeviceQueueCreateInfo> queuesCreateInfo;

    std::set<int> uniqueQueueFamilies{m_indices.graphicsFamily, m_indices.presentingFamily, m_indices.transferFamily};

    for(int queueFamily: uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queuesCreateInfo.push_back(queueCreateInfo);
    }

    // logical device creation
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    if(ENABLE_VALIDATION_LAYERS)
    {
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(debug::VALIDATION_LAYERS.size());
        deviceCreateInfo.ppEnabledLayerNames = debug::VALIDATION_LAYERS.data();
    }
    else
    {
        deviceCreateInfo.enabledLayerCount = 0;
    }

    deviceCreateInfo.pQueueCreateInfos = queuesCreateInfo.data();
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queuesCreateInfo.size());
    deviceCreateInfo.pEnabledFeatures = &m_requiredDeviceFeatures; // Specify device features
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    deviceCreateInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

    VK_CALL(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_logicalDevice));

    vkGetDeviceQueue(m_logicalDevice, m_indices.graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_logicalDevice, m_indices.presentingFamily, 0, &m_presentQueue);
    vkGetDeviceQueue(m_logicalDevice, m_indices.transferFamily, 0, &m_transfertQueue);

    LOG_INFO("Logical device created");
}

void Engine::createSwapChain()
{
    LOG_INFO("Swapchain Creation...");

    m_swapchainDetails = querySwapChainSupport(m_physicalDevice, m_surface);

    chooseSwapSurfaceFormat(m_swapchainDetails.surfaceFormats);
    chooseSwapExtent(m_swapchainDetails.surfaceCapabilities);
    chooseSwapPresentMode(m_swapchainDetails.presentModes);

    uint32_t imageCount = m_swapchainDetails.surfaceCapabilities.minImageCount + 1;

    if(m_swapchainDetails.surfaceCapabilities.maxImageCount > 0 &&
       imageCount > m_swapchainDetails.surfaceCapabilities.maxImageCount)
    {
        imageCount = m_swapchainDetails.surfaceCapabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapChainInfo = {};
    swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainInfo.surface = m_surface;
    swapChainInfo.imageExtent = m_swapchainData.extent;
    swapChainInfo.imageFormat = m_swapchainData.format.format;
    swapChainInfo.imageColorSpace = m_swapchainData.format.colorSpace;
    swapChainInfo.presentMode = m_swapchainData.presentMode;
    swapChainInfo.minImageCount = imageCount;
    swapChainInfo.imageArrayLayers = 1; // Number of layers in the image (different in 3d stereoscopic)
    swapChainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = {static_cast<uint32_t>(m_indices.graphicsFamily),
                                     static_cast<uint32_t>(m_indices.presentingFamily)};

    if(m_indices.graphicsFamily != m_indices.presentingFamily)
    {
        // This line provides us the benefit to share an image in the swapchain across all queue family
        swapChainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapChainInfo.queueFamilyIndexCount = 2;
        swapChainInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else
    {
        swapChainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        swapChainInfo.queueFamilyIndexCount = 0;
        swapChainInfo.pQueueFamilyIndices = nullptr;
    }

    swapChainInfo.preTransform = m_swapchainDetails.surfaceCapabilities.currentTransform;
    swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainInfo.clipped = VK_TRUE;
    swapChainInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CALL(vkCreateSwapchainKHR(m_logicalDevice, &swapChainInfo, nullptr, &m_swapchainData.swapchain));

    vkGetSwapchainImagesKHR(m_logicalDevice, m_swapchainData.swapchain, &imageCount, nullptr);
    m_swapchainData.images.resize(imageCount);
    vkGetSwapchainImagesKHR(m_logicalDevice, m_swapchainData.swapchain, &imageCount, m_swapchainData.images.data());

    m_swapchainData.imageViews.resize(m_swapchainData.images.size());

    // Configure image view for every image in the swapchain
    for(size_t i = 0; i < m_swapchainData.images.size(); i++)
    {
        m_swapchainData.imageViews[i] = utils::createImageView(
          m_logicalDevice, m_swapchainData.format.format, m_swapchainData.images[i], VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    LOG_INFO("Swapchain created");
}

void Engine::createRenderPass()
{
    LOG_INFO("Creating Render Pass...");

    VkAttachmentDescription colorAttachment = {};

    colorAttachment.format = m_swapchainData.format.format;
    colorAttachment.samples = m_msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // Clear the color to constant value at start
    colorAttachment.storeOp =
      VK_ATTACHMENT_STORE_OP_STORE; // Rendered contents will be stored in memory and can be read later
    // We don't use stencil at the moment
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Define which layout we have before render pass
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // and at the end
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = m_msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve = {};
    colorAttachmentResolve.format = m_swapchainData.format.format;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef = {};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    std::array<VkSubpassDescription, 2> subpasses = {};
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[0].colorAttachmentCount = 1;
    // The index of the  attachement is the directive from :
    // layout(location = "index") out vec4 color;
    subpasses[0].pColorAttachments = &colorAttachmentRef;
    subpasses[0].pDepthStencilAttachment = &depthAttachmentRef;
    subpasses[0].pResolveAttachments = &colorAttachmentResolveRef;

    subpasses[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpasses[1].colorAttachmentCount = 1;
    // The index of the  attachement is the directive from :
    // layout(location = "index") out vec4 color;
    subpasses[1].pColorAttachments = &colorAttachmentRef;
    subpasses[1].pDepthStencilAttachment = &depthAttachmentRef;
    subpasses[1].pResolveAttachments = &colorAttachmentResolveRef;

    std::array<VkSubpassDependency, 2> subPassDep = {};
    subPassDep[0].srcSubpass = VK_SUBPASS_EXTERNAL; // Refers to the implicit subpass before the render pass (It it was
                                                    // in dstSubpass would be after the render pass)
    subPassDep[0].dstSubpass = 0;
    subPassDep[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Wait for the swap chain to read the
                                                                                // image
    subPassDep[0].srcAccessMask = 0;
    // The operation which should wait this subpass are read and write operation on color attachement
    subPassDep[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subPassDep[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    subPassDep[1].srcSubpass = 0; // Refers to the implicit subpass before the render pass (It it was
                                  // in dstSubpass would be after the render pass)
    subPassDep[1].dstSubpass = 1;
    subPassDep[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Wait for the swap chain to read the
                                                                                // image
    subPassDep[1].srcAccessMask = 0;
    // The operation which should wait this subpass are read and write operation on color attachement
    subPassDep[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subPassDep[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = subpasses.size();
    renderPassInfo.pSubpasses = subpasses.data();
    renderPassInfo.dependencyCount = subPassDep.size();
    renderPassInfo.pDependencies = subPassDep.data();

    VK_CALL(vkCreateRenderPass(m_logicalDevice, &renderPassInfo, nullptr, &m_renderPass));

    LOG_INFO("Render Pass Created");
}

void Engine::createFramebuffers(VkRenderPass renderPass, const std::vector<VkImageView>& attachements)
{
    LOG_INFO("Creating Framebuffers...");
    m_swapchainData.framebuffers.resize(m_swapchainData.imageViews.size());
    std::vector<VkImageView> frameAttachements;
    frameAttachements.insert(frameAttachements.begin(), attachements.begin(), attachements.end());

    for(size_t i = 0; i < m_swapchainData.imageViews.size(); i++)
    {
        frameAttachements.push_back(m_swapchainData.imageViews[i]);
        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(frameAttachements.size());
        framebufferInfo.pAttachments = frameAttachements.data();
        framebufferInfo.layers = 1;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.height = m_swapchainData.extent.height;
        framebufferInfo.width = m_swapchainData.extent.width;

        VK_CALL(vkCreateFramebuffer(m_logicalDevice, &framebufferInfo, nullptr, &m_swapchainData.framebuffers[i]));

        frameAttachements.pop_back();
    }

    LOG_INFO("Framebuffers Created");
}

void Engine::createDescriptorSetLayout()
{
    LOG_INFO("Creating Descriptor Set Layout...");
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0; // Value in shader "layout(binding = 0) uniform"
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1; // Number of object to pass
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1; // Number of object to pass
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VK_CALL(vkCreateDescriptorSetLayout(m_logicalDevice, &layoutInfo, nullptr, &m_descriptorSetLayout));

    LOG_INFO("Descriptor Set Layout Created");
}

void Engine::createGraphicsPipeline()
{
    LOG_INFO("Creating Graphics Pipeline...");
    auto vertexShader = readFile(std::string(SHADER_PATH) + "/vertex.spv");
    auto fragmentShader = readFile(std::string(SHADER_PATH) + "/fragment.spv");

    VkShaderModule vertexShaderModule = createShaderModule(vertexShader);
    VkShaderModule fragmentShaderModule = createShaderModule(fragmentShader);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.module = vertexShaderModule;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.pName = "main";
    vertShaderStageInfo.pSpecializationInfo = nullptr; // Can configure constant values in shader code,
    // which is more optimized than have a configurable value at runtime

    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.module = fragmentShaderModule;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStageInfos[] = {vertShaderStageInfo, fragShaderStageInfo};

    // describe the infos inputed for the vertex and the structure of the datas (size, offset,...)
    auto vertexBindingDescription = descriptor::getVertexBindingDescription();
    auto vertexAttributeDescriptions = descriptor::getVertexAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &vertexBindingDescription;
    vertexInputInfo.vertexBindingDescriptionCount = 1;

    VkPipelineInputAssemblyStateCreateInfo assemblyInfos = {};
    assemblyInfos.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assemblyInfos.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    assemblyInfos.primitiveRestartEnable = VK_FALSE;

    m_viewport = {};
    m_viewport.x = 0;
    m_viewport.y = 0;
    m_viewport.width = static_cast<float>(m_swapchainData.extent.width);
    m_viewport.height = static_cast<float>(m_swapchainData.extent.height);
    m_viewport.minDepth = 0.0f;
    m_viewport.maxDepth = 1.0f;

    // The scissor is masking the "out of the scissor rectangle" data from the viewport
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainData.extent;

    VkPipelineViewportStateCreateInfo viewPortInfo = {};
    viewPortInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewPortInfo.viewportCount = 1;
    viewPortInfo.pScissors = &scissor;
    viewPortInfo.scissorCount = 1;
    viewPortInfo.pViewports = &m_viewport;

    VkPipelineRasterizationStateCreateInfo rasterizerInfo = {};
    rasterizerInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerInfo.lineWidth = 1.0f;
    rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerInfo.frontFace = VK_FRONT_FACE_CLOCKWISE; // VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizerInfo.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multiSampInfo = {};
    multiSampInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multiSampInfo.sampleShadingEnable = VK_TRUE;
    multiSampInfo.rasterizationSamples = m_msaaSamples;
    multiSampInfo.minSampleShading = 0.2f;

    VkPipelineDepthStencilStateCreateInfo stencilInfos = {};
    stencilInfos.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    stencilInfos.depthCompareOp = VK_COMPARE_OP_LESS;
    stencilInfos.depthTestEnable = VK_TRUE;
    stencilInfos.depthWriteEnable = VK_TRUE;

    // Color blend describe how we want to replace the color in the current framebuffer
    VkPipelineColorBlendAttachmentState colorBlend = {};
    colorBlend.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlend.blendEnable = VK_FALSE;
    colorBlend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlend.colorBlendOp = VK_BLEND_OP_ADD; // Seems to have pretty cool fast features here
    colorBlend.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlendInfo = {};
    colorBlendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendInfo.pAttachments = &colorBlend;
    colorBlendInfo.logicOpEnable = VK_FALSE; // Use for bitwise operation
    colorBlendInfo.logicOp = VK_LOGIC_OP_COPY;
    colorBlendInfo.attachmentCount = 1;
    // colorBlendInfo.blendConstants[0...4] = floatValue;

    // This set which of the previous value can be dynamically change during runtime !!!
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_LINE_WIDTH};

    VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
    dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.pDynamicStates = dynamicStates;
    dynamicStateInfo.dynamicStateCount = 1;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    VK_CALL(vkCreatePipelineLayout(m_logicalDevice, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStageInfos;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.pMultisampleState = &multiSampInfo;
    pipelineInfo.pColorBlendState = &colorBlendInfo;
    pipelineInfo.pDepthStencilState = &stencilInfos;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.pInputAssemblyState = &assemblyInfos;
    pipelineInfo.pRasterizationState = &rasterizerInfo;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pViewportState = &viewPortInfo;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.subpass = 0;
    // We can make pipeline derivates from other pipelines if they have similarities
    // We have a single pipeline here, so we don't use it
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if(vkCreateGraphicsPipelines(m_logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) !=
       VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(m_logicalDevice, vertexShaderModule, nullptr);
    vkDestroyShaderModule(m_logicalDevice, fragmentShaderModule, nullptr);

    LOG_INFO("Graphics Pipeline Created");
}

void Engine::createCommandPool()
{
    LOG_INFO("Creating Command Pools...");

    // Create a command pool only for graphics operations
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = m_indices.graphicsFamily;
    commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CALL(vkCreateCommandPool(m_logicalDevice, &commandPoolInfo, nullptr, &m_commandPool));

    if(m_indices.transferAvailable())
    {
        commandPoolInfo.queueFamilyIndex = m_indices.transferFamily;

        VK_CALL(vkCreateCommandPool(m_logicalDevice, &commandPoolInfo, nullptr, &m_commandPoolTransfert));
    }

    LOG_INFO("Command Pools Created");
}

void Engine::createDepthRessources()
{
    VkFormat depthFormat = findDepthFormat();
    utils::createImage(m_logicalDevice,
                       m_swapchainData.extent.width,
                       m_swapchainData.extent.height,
                       1,
                       m_msaaSamples,
                       depthFormat,
                       VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       m_memoryProperties,
                       m_depthImage,
                       m_depthImageMemory);

    m_depthImageView = utils::createImageView(m_logicalDevice, depthFormat, m_depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    utils::transitionImageLayout(m_logicalDevice,
                                 getCommandPool(),
                                 getGraphicsQueue(),
                                 m_depthImage,
                                 depthFormat,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 1);
}

void Engine::createColorRessources()
{
    VkFormat format = m_swapchainData.format.format;

    utils::createImage(m_logicalDevice,
                       m_swapchainData.extent.width,
                       m_swapchainData.extent.height,
                       1,
                       m_msaaSamples,
                       format,
                       VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       m_memoryProperties,
                       m_colorImage,
                       m_colorMemory);
    m_colorImageView = utils::createImageView(m_logicalDevice, format, m_colorImage, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    utils::transitionImageLayout(m_logicalDevice,
                                 getCommandPool(),
                                 getGraphicsQueue(),
                                 m_colorImage,
                                 format,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 1);
}
void Engine::initImgui()
{
    {
        VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
                                             {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
                                             {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
                                             {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
                                             {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
                                             {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
                                             {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = static_cast<uint32_t>(IM_ARRAYSIZE(pool_sizes));
        pool_info.pPoolSizes = pool_sizes;
        VK_CALL(vkCreateDescriptorPool(m_logicalDevice, &pool_info, nullptr, &m_imguiDescriptorPool));
    }
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = m_instance;
    init_info.PhysicalDevice = m_physicalDevice;
    init_info.Device = m_logicalDevice;
    init_info.QueueFamily = m_indices.graphicsFamily;
    init_info.Queue = getGraphicsQueue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = m_imguiDescriptorPool;
    init_info.Subpass = 1;
    init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
    init_info.ImageCount = m_swapchainData.images.size();
    init_info.MSAASamples = getMaxUsableSampleCount();
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = nullptr;
    ImGui_ImplVulkan_Init(&init_info, m_renderPass);
    // Upload Fonts
    {
        // Use any command queue
        auto command = utils::beginSingleTimeCommands(m_logicalDevice, getCommandPool());
        ImGui_ImplVulkan_CreateFontsTexture(command);
        utils::endSingleTimeCommands(m_logicalDevice, getCommandPool(), getGraphicsQueue(), command);
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }
}

void Engine::recreateCommandBuffer()
{
    vkDeviceWaitIdle(m_logicalDevice);
    vkFreeCommandBuffers(
      m_logicalDevice, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()), m_commandBuffers.data());
    vkFreeCommandBuffers(m_logicalDevice,
                         m_commandPool,
                         static_cast<uint32_t>(m_commandBufferSecondary.size()),
                         m_commandBufferSecondary.data());
    createPrimaryCommandBuffer();
    createSecondaryCommandBuffers();
}

void Engine::createVertexBuffer()
{
    LOG_INFO("Creating and Allocating Vertex Buffer");
    VkDeviceSize bufferSize = sizeof(m_mesh.vertices[0]) * m_mesh.vertices.size();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(m_logicalDevice,
                        m_indices,
                        bufferSize,
                        usage,
                        properties,
                        m_memoryProperties,
                        stagingBuffer,
                        stagingBufferMemory);

    void* pData; // Contains a pointer to the mapped memory

    // Documentation : memory must have been created with a memory type that reports VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    //                  flags is reserved for future use of the vulkanAPI.
    vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &pData);
    memcpy(pData, m_mesh.vertices.data(), static_cast<std::size_t>(bufferSize));
    vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

    utils::createBuffer(m_logicalDevice,
                        m_indices,
                        bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        m_memoryProperties,
                        m_meshData.vertexBuffer,
                        m_meshData.vertexBufferMemory);
    utils::copyBuffer(m_logicalDevice,
                      getCommandPoolTransfer(),
                      getTransfertQueue(),
                      stagingBuffer,
                      m_meshData.vertexBuffer,
                      bufferSize);

    vkDestroyBuffer(m_logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(m_logicalDevice, stagingBufferMemory, nullptr);
    LOG_INFO("Vertex Buffer Created");
}

void Engine::createVertexIndexBuffer()
{
    LOG_INFO("Creating and Allocating Index Buffer");
    VkDeviceSize bufferSize = sizeof(m_mesh.faces[0]) * m_mesh.faces.size();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(m_logicalDevice,
                        m_indices,
                        bufferSize,
                        usage,
                        properties,
                        m_memoryProperties,
                        stagingBuffer,
                        stagingBufferMemory);

    void* pData; // Contains a pointer to the mapped memory

    // Documentation : memory must have been created with a memory type that reports VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    //                  flags is reserved for future use of the vulkanAPI.
    vkMapMemory(m_logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &pData);
    memcpy(pData, m_mesh.faces.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(m_logicalDevice, stagingBufferMemory);

    utils::createBuffer(m_logicalDevice,
                        m_indices,
                        bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        m_memoryProperties,
                        m_meshData.indexBuffer,
                        m_meshData.indexBufferMemory);
    utils::copyBuffer(m_logicalDevice,
                      getCommandPoolTransfer(),
                      getTransfertQueue(),
                      stagingBuffer,
                      m_meshData.indexBuffer,
                      bufferSize);

    vkDestroyBuffer(m_logicalDevice, stagingBuffer, nullptr);
    vkFreeMemory(m_logicalDevice, stagingBufferMemory, nullptr);
    LOG_INFO("Index Buffer Created");
}

void Engine::createUniformBuffer()
{
    LOG_INFO("Creating Uniform Buffer...");
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    m_uniformBuffers.resize(m_swapchainData.imageViews.size());
    m_uniformBuffersMemory.resize(m_swapchainData.imageViews.size());

    for(size_t i = 0; i < m_swapchainData.imageViews.size(); i++)
    {
        utils::createBuffer(m_logicalDevice,
                            m_indices,
                            bufferSize,
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            m_memoryProperties,
                            m_uniformBuffers[i],
                            m_uniformBuffersMemory[i]);
    }

    LOG_INFO("Uniform Buffer Created");
}

void Engine::updateUniformBuffer(uint32_t imageIndex)
{
    uint16_t angle = 0;

    UniformBufferObject ubo = {};

    ubo.model = Matrix4::Identity();

    ubo.view = m_camera->getView();

    ubo.projection = m_camera->getProjection();

    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = 2 * std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    ubo.lightPos = Vector3(4 * cos(time), 3, 4 * sin(time));

    void* pData;

    for(uint8_t i = 0; i < m_swapchainData.imageViews.size(); i++)
    {
        vkMapMemory(m_logicalDevice, m_uniformBuffersMemory[i], 0, sizeof(UniformBufferObject), 0, &pData);
        memcpy(pData, &ubo, sizeof(UniformBufferObject));
        vkUnmapMemory(m_logicalDevice, m_uniformBuffersMemory[i]);
    }
}

void Engine::createDescriptorPool()
{
    LOG_INFO("Creating Descriptor Pool...");

    std::array<VkDescriptorPoolSize, 2> descPoolSizes = {};
    // UBO
    descPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descPoolSizes[0].descriptorCount = static_cast<uint32_t>(m_swapchainData.images.size());

    // Textures
    descPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descPoolSizes[1].descriptorCount = static_cast<uint32_t>(m_swapchainData.images.size());

    VkDescriptorPoolCreateInfo descPoolInfo = {};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = static_cast<uint32_t>(descPoolSizes.size());
    descPoolInfo.pPoolSizes = descPoolSizes.data();
    descPoolInfo.maxSets = static_cast<uint32_t>(m_swapchainData.images.size());

    VK_CALL(vkCreateDescriptorPool(m_logicalDevice, &descPoolInfo, nullptr, &m_descriptorPool));

    LOG_INFO("Descriptor Pool Created");
}

void Engine::createDescriptorSets()
{
    LOG_INFO("Creating Descriptor Sets...");
    std::vector<VkDescriptorSetLayout> layouts(m_swapchainData.images.size(), m_descriptorSetLayout);
    VkDescriptorSetAllocateInfo descAlloc = {};
    descAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAlloc.descriptorPool = m_descriptorPool;
    descAlloc.descriptorSetCount = static_cast<uint32_t>(m_swapchainData.images.size());
    descAlloc.pSetLayouts = layouts.data();

    m_descriptorSets.resize(m_swapchainData.images.size());

    VK_CALL(vkAllocateDescriptorSets(m_logicalDevice, &descAlloc, m_descriptorSets.data()));

    for(size_t i = 0; i < m_swapchainData.images.size(); i++)
    {
        VkDescriptorBufferInfo descBufferInfo = {};
        descBufferInfo.buffer = m_uniformBuffers[i];
        descBufferInfo.offset = 0;
        descBufferInfo.range = sizeof(UniformBufferObject);

        // VkDescriptorImageInfo imageInfo = {};
        // imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // imageInfo.imageView = m_lenaTexture.getImageView();
        // imageInfo.sampler = m_lenaTexture.getSampler();

        std::array<VkWriteDescriptorSet, 1> writeInfos = {};
        writeInfos[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeInfos[0].dstSet = m_descriptorSets[i];
        writeInfos[0].dstBinding = 0; // binding index in "layout(binding = 0)"
        writeInfos[0].dstArrayElement = 0;
        writeInfos[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeInfos[0].descriptorCount = 1; // We can update multiple descriptor at once in an array
        writeInfos[0].pBufferInfo = &descBufferInfo;

        // writeInfos[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeInfos[1].dstSet = m_descriptorSets[i];
        // writeInfos[1].dstBinding = 1; // binding index in "layout(binding = 0)"
        // writeInfos[1].dstArrayElement = 0;
        // writeInfos[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        // writeInfos[1].descriptorCount = 1; // We can update multiple descriptor at once in an array
        // writeInfos[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(
          m_logicalDevice, static_cast<uint32_t>(writeInfos.size()), writeInfos.data(), 0, nullptr);
    }

    LOG_INFO("Descriptor Sets Created");
}

/*@brief :  Create the command buffers associated with the command pool
 *           We have a command buffer for every image views in the swapchain
 *           to render each one of them
 */
void Engine::createSecondaryCommandBuffers()
{
    LOG_INFO("Creating and Recording Command Buffers...");

    // Secondary command buffer
    m_commandBufferSecondary.resize(m_swapchainData.framebuffers.size());
    VkCommandBufferAllocateInfo allocateBufferInfo = {};
    allocateBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateBufferInfo.commandPool = m_commandPool;
    allocateBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocateBufferInfo.commandBufferCount = m_commandBufferSecondary.size();

    VK_CALL(vkAllocateCommandBuffers(m_logicalDevice, &allocateBufferInfo, m_commandBufferSecondary.data()));
    VkCommandBufferBeginInfo commandBeginInfo = {};
    commandBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBeginInfo.flags =
      VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    for(std::size_t i = 0; i < m_commandBuffers.size(); ++i)
    {
        VkCommandBufferInheritanceInfo inheritanceInfo{};
        inheritanceInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inheritanceInfo.framebuffer = m_swapchainData.framebuffers[i];
        inheritanceInfo.renderPass = m_renderPass;
        inheritanceInfo.subpass = 0;
        inheritanceInfo.occlusionQueryEnable = VK_FALSE;
        inheritanceInfo.queryFlags = 0;
        inheritanceInfo.pipelineStatistics = 0;
        commandBeginInfo.pInheritanceInfo = &inheritanceInfo;

        VK_CALL(vkBeginCommandBuffer(m_commandBufferSecondary[i], &commandBeginInfo));

        {
            vkCmdSetViewport(m_commandBufferSecondary[i], 0, 1, &m_viewport);
            vkCmdBindPipeline(m_commandBufferSecondary[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

            VkBuffer vertexBuffers[] = {m_meshData.vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(m_commandBufferSecondary[i], 0, 1, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(m_commandBufferSecondary[i], m_meshData.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(m_commandBufferSecondary[i],
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_pipelineLayout,
                                    0,
                                    1,
                                    &m_descriptorSets[0],
                                    0,
                                    nullptr);

            // 1 used for the instanced rendering could be higher i think for multiple instanced
            vkCmdDrawIndexed(m_commandBufferSecondary[i], static_cast<uint32_t>(m_mesh.faces.size() * 3), 1, 0, 0, 0);
        }
        VK_CALL(vkEndCommandBuffer(m_commandBufferSecondary[i]));
    }

    LOG_INFO("Command Buffers Created");
}

void Engine::createPrimaryCommandBuffer()
{
    m_commandBuffers.resize(m_swapchainData.framebuffers.size());
    VkCommandBufferAllocateInfo allocateBufferInfo = {};
    allocateBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateBufferInfo.commandPool = m_commandPool;
    allocateBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Can be submitted to queue, and SECONDARY can be
                                                                // called inside a primary command buffer
    allocateBufferInfo.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());

    VK_CALL(vkAllocateCommandBuffers(m_logicalDevice, &allocateBufferInfo, m_commandBuffers.data()));
}

void Engine::updatePrimaryCommandBuffers()
{
    for(std::size_t i = 0; i < m_commandBuffers.size(); ++i)
    {
        VkCommandBufferBeginInfo commandBeginInfo = {};
        commandBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        commandBeginInfo.pInheritanceInfo = nullptr;

        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].color = {1.0f, (153.0f / 255.0f), (51.0f / 255.0f), 1.0f};
        clearValues[1].depthStencil = {1.0, 0};

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.renderPass = m_renderPass;
        renderPassBeginInfo.framebuffer = m_swapchainData.framebuffers[i];
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderArea.offset.x = 0;
        renderPassBeginInfo.renderArea.offset.y = 0;
        renderPassBeginInfo.renderArea.extent = m_swapchainData.extent;
        renderPassBeginInfo.clearValueCount = clearValues.size();
        renderPassBeginInfo.pClearValues = clearValues.data();

        VK_CALL(vkBeginCommandBuffer(m_commandBuffers[i], &commandBeginInfo));
        {
            vkCmdBeginRenderPass(
              m_commandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);

            vkCmdExecuteCommands(m_commandBuffers[i], 1, &m_commandBufferSecondary[i]);

            vkCmdNextSubpass(m_commandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);
            ImDrawData* draw_data = ImGui::GetDrawData();
            const bool is_minimized = (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f);
            if(!is_minimized)
            {
                ImGui_ImplVulkan_RenderDrawData(draw_data, m_commandBuffers[i]);
            }

            vkCmdEndRenderPass(m_commandBuffers[i]);
        }

        VK_CALL(vkEndCommandBuffer(m_commandBuffers[i]));
    }
}

void Engine::createSyncObjects()
{
    LOG_INFO("Creating Synchronization Objects...");
    m_imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VK_CALL(vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_imageAvailableSemaphore[i]));
        VK_CALL(vkCreateSemaphore(m_logicalDevice, &semaphoreInfo, nullptr, &m_renderFinishedSemaphore[i]));
        VK_CALL(vkCreateFence(m_logicalDevice, &fenceInfo, nullptr, &m_inFlightFences[i]));
    }
    LOG_INFO("Synchronization Objects Created");
}

void Engine::checkApplicationState()
{
    for(size_t idx = 0; idx < sizeof(ApplicationStateChange); idx += sizeof(bool))
    {
        bool* state = (reinterpret_cast<bool*>(&m_applicationChanges) + idx);

        if(state == &m_applicationChanges.modelModified && *state)
        {
            recreateCommandBuffer();
            *state = false;
            return;
        }
        else if(state == &m_applicationChanges.modelModified && state)
        {
            return;
        }
    }
}

std::vector<char> Engine::readFile(const std::string& fileName)
{
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);

    if(!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkShaderModule Engine::createShaderModule(const std::vector<char>& shaderCode)
{
    LOG_INFO("Creating Shader Modules...");
    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = shaderCode.size();
    shaderInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule;

    VK_CALL(vkCreateShaderModule(m_logicalDevice, &shaderInfo, nullptr, &shaderModule));

    LOG_INFO("Shader Module Created");

    return shaderModule;
}

VkSampleCountFlagBits Engine::getMaxUsableSampleCount()
{
    VkSampleCountFlags counts = std::min(m_deviceProperties.limits.framebufferColorSampleCounts,
                                         m_deviceProperties.limits.framebufferDepthSampleCounts);

    if(counts & VK_SAMPLE_COUNT_32_BIT)
        return VK_SAMPLE_COUNT_32_BIT;

    if(counts & VK_SAMPLE_COUNT_16_BIT)
        return VK_SAMPLE_COUNT_16_BIT;

    if(counts & VK_SAMPLE_COUNT_8_BIT)
        return VK_SAMPLE_COUNT_8_BIT;

    if(counts & VK_SAMPLE_COUNT_4_BIT)
        return VK_SAMPLE_COUNT_4_BIT;

    if(counts & VK_SAMPLE_COUNT_2_BIT)
        return VK_SAMPLE_COUNT_2_BIT;

    return VK_SAMPLE_COUNT_1_BIT;
}

void Engine::recreateSwapChain()
{
    vkDeviceWaitIdle(m_logicalDevice);

    cleanUpSwapChain();
    createSwapChain();
    createRenderPass();
    createGraphicsPipeline();
    createDepthRessources();
    createColorRessources();
    createFramebuffers(m_renderPass, {m_colorImageView, m_depthImageView});
    createPrimaryCommandBuffer();
    createSecondaryCommandBuffers();
}

void Engine::cleanUpSwapChain()
{
    for(auto framebuffer: m_swapchainData.framebuffers)
    {
        vkDestroyFramebuffer(m_logicalDevice, framebuffer, nullptr);
    }

    vkFreeCommandBuffers(
      m_logicalDevice, m_commandPool, m_commandBufferSecondary.size(), m_commandBufferSecondary.data());
    vkFreeCommandBuffers(m_logicalDevice, m_commandPool, m_commandBuffers.size(), m_commandBuffers.data());

    vkDestroyImageView(m_logicalDevice, m_depthImageView, nullptr);
    vkDestroyImage(m_logicalDevice, m_depthImage, nullptr);
    vkFreeMemory(m_logicalDevice, m_depthImageMemory, nullptr);

    vkDestroyImageView(m_logicalDevice, m_colorImageView, nullptr);
    vkDestroyImage(m_logicalDevice, m_colorImage, nullptr);
    vkFreeMemory(m_logicalDevice, m_colorMemory, nullptr);

    vkDestroyPipeline(m_logicalDevice, m_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_logicalDevice, m_pipelineLayout, nullptr);
    vkDestroyRenderPass(m_logicalDevice, m_renderPass, nullptr);

    for(auto imageView: m_swapchainData.imageViews)
    {
        vkDestroyImageView(m_logicalDevice, imageView, nullptr);
    }

    vkDestroySwapchainKHR(m_logicalDevice, m_swapchainData.swapchain, nullptr);
}

void Engine::cleanup()
{
    if(!m_isCleaned)
    {
        m_isCleaned = true;
        cleanUpSwapChain();

        // Descriptor Set/Pool
        vkDestroyDescriptorPool(m_logicalDevice, m_descriptorPool, nullptr);
        vkDestroyDescriptorPool(m_logicalDevice, m_imguiDescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(m_logicalDevice, m_descriptorSetLayout, nullptr);

        // m_lenaTexture.destroy();

        // Vertex/Uniform/Index buffers
        for(size_t i = 0; i < m_swapchainData.images.size(); i++)
        {
            vkDestroyBuffer(m_logicalDevice, m_uniformBuffers[i], nullptr);
            vkFreeMemory(m_logicalDevice, m_uniformBuffersMemory[i], nullptr);
        }

        // Mesh data
        vkDestroyBuffer(m_logicalDevice, m_meshData.vertexBuffer, nullptr);
        vkFreeMemory(m_logicalDevice, m_meshData.vertexBufferMemory, nullptr);
        vkDestroyBuffer(m_logicalDevice, m_meshData.indexBuffer, nullptr);
        vkFreeMemory(m_logicalDevice, m_meshData.indexBufferMemory, nullptr);

        // Semaphores
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(m_logicalDevice, m_imageAvailableSemaphore[i], nullptr);
            vkDestroySemaphore(m_logicalDevice, m_renderFinishedSemaphore[i], nullptr);
            vkDestroyFence(m_logicalDevice, m_inFlightFences[i], nullptr);
        }

        // Command Pool
        vkDestroyCommandPool(m_logicalDevice, m_commandPool, nullptr);

        if(m_indices.transferAvailable())
        {
            vkDestroyCommandPool(m_logicalDevice, m_commandPoolTransfert, nullptr);
        }

        vkDestroyDevice(m_logicalDevice, nullptr);
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

        if constexpr(ENABLE_VALIDATION_LAYERS)
        {
            debug::destroyDebugMessenger(m_instance, m_debugMessenger);
        }

        vkDestroyInstance(m_instance, nullptr);
    }
}

/*@brief : Choose the optimal surface format for the swap chain and return it
 */
void Engine::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    if(availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
    {
        m_swapchainData.format = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        return;
    }

    for(const auto& format: availableFormats)
    {
        if(format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            m_swapchainData.format = format;
            return;
        }
    }

    m_swapchainData.format = availableFormats[0];
}

/*@brief : Choose the optimal present mode for the swap chain and return it
 */
void Engine::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes)
{
    VkPresentModeKHR bestmode = VK_PRESENT_MODE_FIFO_KHR;

    for(const auto& mode: availableModes)
    {
        if(mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            m_swapchainData.presentMode = mode;
            return;
        }
        else if(mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            bestmode = mode;
        }
    }

    m_swapchainData.presentMode = bestmode;
}

void Engine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        m_swapchainData.extent = capabilities.currentExtent;
        return;
    }
    else
    {
        VkExtent2D actualExtent = m_windowExtent;
        actualExtent.width =
          std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));
        m_swapchainData.extent = actualExtent;
    }
}

} // namespace engine

namespace
{
bool isDeviceSuitable(VkPhysicalDevice device,
                      VkSurfaceKHR surface,
                      const std::vector<const char*>& extensions,
                      const VkPhysicalDeviceFeatures& requiredFeatures)
{
    VkPhysicalDeviceProperties deviceProperties = {};
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    VkPhysicalDeviceFeatures deviceFeatures = {};
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    engine::QueueFamilyIndices indices = engine::findQueueFamilies(device, surface);

    bool isDeviceExtensionsSupported = engine::checkDeviceExtensionSupport(device, extensions);

    bool swapChainAdequate = false;

    if(isDeviceExtensionsSupported)
    {
        engine::SwapchainSupportDetails swapChainSupport = engine::querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.surfaceFormats.empty() && !swapChainSupport.presentModes.empty();
    }

    return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
           engine::isDeviceContainingFeatures(deviceFeatures, requiredFeatures) && isDeviceExtensionsSupported &&
           swapChainAdequate && indices.isComplete();
}
VkPhysicalDevice getBestPhysicalDevice(VkInstance instance,
                                       VkSurfaceKHR surface,
                                       const std::vector<const char*>& extensions,
                                       const VkPhysicalDeviceFeatures& requiredFeatures)
{
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;

    uint32_t deviceCount = 0;

    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if(deviceCount == 0)
    {
        LOG_ERROR("No GPU found compatible with vulkan");
        throw std::runtime_error("No GPU found compatible with vulkan!");
    }

    std::vector<VkPhysicalDevice> availableDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, availableDevices.data());

    for(const auto& device: availableDevices)
    {
        if(isDeviceSuitable(device, surface, extensions, requiredFeatures))
        {
            bestDevice = device;
            break;
        }
    }

    if(bestDevice == VK_NULL_HANDLE)
    {
        LOG_ERROR("Failed to find a suitable GPU!");
        throw std::runtime_error("Failed to find a suitable GPU!");
    }

    return bestDevice;
}
} // namespace
