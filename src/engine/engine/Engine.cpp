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
#include <set>

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
    requiredDeviceFeatures_.sampleRateShading = VK_TRUE;
    auto scene = IO::loadScene(MESH_PATH);
    if(!scene)
    {
        THROW(std::runtime_error("Could not load the mesh"));
    }
    mesh_ = scene->meshes[0];
    camera_ = std::move(scene->cameras.front());
    if constexpr(ENABLE_VALIDATION_LAYERS)
    {
        requiredExtensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
}

void Engine::drawFrame()
{
    checkApplicationState();

    uint32_t imageIndex;

    vkWaitForFences(logicalDevice_, 1, &inFlightFences_[currentFrame_], VK_TRUE, std::numeric_limits<uint64_t>::max());
    vkResetFences(logicalDevice_, 1, &inFlightFences_[currentFrame_]);
    // Timeout in nanoseconds = numeric_limits... using the max disable the timeout
    VkResult result = vkAcquireNextImageKHR(logicalDevice_,
                                            swapchainData_.swapchain,
                                            std::numeric_limits<uint64_t>::max(),
                                            imageAvailableSemaphore_[currentFrame_],
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

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAvailableSemaphore_[currentFrame_];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphore_[currentFrame_];

    VkPipelineStageFlags waitStageFlags[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.pWaitDstStageMask = waitStageFlags; // We wait the semaphore with to have the entry with the same index
    // Here we wait the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT of the imageAvailableSemaphore_

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];

    vkResetFences(logicalDevice_, 1, &inFlightFences_[currentFrame_]);

    VK_CALL(vkQueueSubmit(presentQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]));

    VkResult presentatioResult;
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchainData_.swapchain;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore_[currentFrame_];
    presentInfo.pResults = &presentatioResult; // Array of results for each swap chain images

    result = vkQueuePresentKHR(presentQueue_, &presentInfo);

    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResize)
    {
        framebufferResize = false;
        recreateSwapChain();
    }
    else if(result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    vkQueueWaitIdle(presentQueue_);

    currentFrame_ = (currentFrame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

Engine::~Engine()
{
    cleanup();
}

void Engine::addRequiredExtensions(const char** extensions, uint32_t extensionCount)
{
    requiredExtensions_.insert(requiredExtensions_.end(), extensions, extensions + extensionCount);
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
    createFramebuffers(renderPass_, {colorImageView_, depthImageView_});
    // lenaTexture_.create();
    createVertexBuffer();
    createVertexIndexBuffer();
    createUniformBuffer();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();

    LOG_INFO("Vulkan Initialisation Finished");
}

void Engine::setSurface(const VkSurfaceKHR& surface)
{
    surface_ = surface;
}

VkQueue Engine::getTransfertQueue() const
{
    if(indices_.transferAvailable())
    {
        return transfertQueue_;
    }
    return getGraphicsQueue();
}

VkQueue Engine::getGraphicsQueue() const
{
    return graphicsQueue_;
}

VkCommandPool Engine::getCommandPoolTransfer() const
{
    if(indices_.transferAvailable())
    {
        return commandPoolTransfert_;
    }
    return getCommandPool();
}

VkCommandPool Engine::getCommandPool() const
{
    return commandPool_;
}

void Engine::createInstance()
{
    isCleaned_ = false;

    // check if the validation layers are needed and if they are available
    if(ENABLE_VALIDATION_LAYERS && !debug::checkValidationLayerSupport())
    {
        THROW(std::runtime_error("Validation layer requested, but not available !"));
    }

    VK_CALL(
      areInstanceExtensionsCompatible(requiredExtensions_.data(), static_cast<uint32_t>(requiredExtensions_.size())));

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

    createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions_.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions_.data();

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);

    VK_CALL(result);

    if constexpr(ENABLE_VALIDATION_LAYERS)
    {
        debug::createDebugMessenger(instance_, debugMessenger_, debug::defaultDebugCallback);
    }

    LOG_INFO("Vulkan Instance Created");
}

VkInstance Engine::getInstance()
{
    return instance_;
}

void Engine::resizeExtent(int width, int height)
{
    if(windowExtent_.width == width && windowExtent_.height == height)
    {
        return;
    }
    framebufferResize = true;
    windowExtent_.width = width;
    windowExtent_.height = height;
    camera_->setViewportDimensions(width, height);
    swapchainDetails_ = querySwapChainSupport(physicalDevice_, surface_); // Used for querySwapChainSupport
}

void Engine::setCamera(std::shared_ptr<Camera> camera)
{
    camera_ = std::move(camera);
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
    return findSupportedTilingFormat(physicalDevice_,
                                     {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                                     VK_IMAGE_TILING_OPTIMAL,
                                     VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void Engine::pickPhysicalDevice()
{
    LOG_INFO("Picking a physical device");

    physicalDevice_ = getBestPhysicalDevice(instance_, surface_, DEVICE_EXTENSIONS, requiredDeviceFeatures_);

    vkGetPhysicalDeviceProperties(physicalDevice_, &deviceProperties_);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
    msaaSamples_ = getMaxUsableSampleCount();
    LOG_INFO(deviceProperties_.deviceName << " : I chose you !!!!");
}

void Engine::createLogicalDevice()
{
    LOG_INFO("Creating a logical device...");
    indices_ = findQueueFamilies(physicalDevice_, surface_);
    float queuePriority = 1.0f;

    // Specifying the queues to be created
    std::vector<VkDeviceQueueCreateInfo> queuesCreateInfo;

    std::set<int> uniqueQueueFamilies{indices_.graphicsFamily, indices_.presentingFamily, indices_.transferFamily};

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
    deviceCreateInfo.pEnabledFeatures = &requiredDeviceFeatures_; // Specify device features
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
    deviceCreateInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

    VK_CALL(vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &logicalDevice_));

    vkGetDeviceQueue(logicalDevice_, indices_.graphicsFamily, 0, &graphicsQueue_);
    vkGetDeviceQueue(logicalDevice_, indices_.presentingFamily, 0, &presentQueue_);
    vkGetDeviceQueue(logicalDevice_, indices_.transferFamily, 0, &transfertQueue_);

    LOG_INFO("Logical device created");
}

void Engine::createSwapChain()
{
    LOG_INFO("Swapchain Creation...");

    swapchainDetails_ = querySwapChainSupport(physicalDevice_, surface_);

    chooseSwapSurfaceFormat(swapchainDetails_.surfaceFormats);
    chooseSwapExtent(swapchainDetails_.surfaceCapabilities);
    chooseSwapPresentMode(swapchainDetails_.presentModes);

    uint32_t imageCount = swapchainDetails_.surfaceCapabilities.minImageCount + 1;

    if(swapchainDetails_.surfaceCapabilities.maxImageCount > 0 &&
       imageCount > swapchainDetails_.surfaceCapabilities.maxImageCount)
    {
        imageCount = swapchainDetails_.surfaceCapabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapChainInfo = {};
    swapChainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapChainInfo.surface = surface_;
    swapChainInfo.imageExtent = swapchainData_.extent;
    swapChainInfo.imageFormat = swapchainData_.format.format;
    swapChainInfo.imageColorSpace = swapchainData_.format.colorSpace;
    swapChainInfo.presentMode = swapchainData_.presentMode;
    swapChainInfo.minImageCount = imageCount;
    swapChainInfo.imageArrayLayers = 1; // Number of layers in the image (different in 3d stereoscopic)
    swapChainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = {static_cast<uint32_t>(indices_.graphicsFamily),
                                     static_cast<uint32_t>(indices_.presentingFamily)};

    if(indices_.graphicsFamily != indices_.presentingFamily)
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

    swapChainInfo.preTransform = swapchainDetails_.surfaceCapabilities.currentTransform;
    swapChainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapChainInfo.clipped = VK_TRUE;
    swapChainInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CALL(vkCreateSwapchainKHR(logicalDevice_, &swapChainInfo, nullptr, &swapchainData_.swapchain));

    vkGetSwapchainImagesKHR(logicalDevice_, swapchainData_.swapchain, &imageCount, nullptr);
    swapchainData_.images.resize(imageCount);
    vkGetSwapchainImagesKHR(logicalDevice_, swapchainData_.swapchain, &imageCount, swapchainData_.images.data());

    swapchainData_.imageViews.resize(swapchainData_.images.size());

    // Configure image view for every image in the swapchain
    for(size_t i = 0; i < swapchainData_.images.size(); i++)
    {
        swapchainData_.imageViews[i] = utils::createImageView(
          logicalDevice_, swapchainData_.format.format, swapchainData_.images[i], VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    LOG_INFO("Swapchain created");
}

void Engine::createRenderPass()
{
    LOG_INFO("Creating Render Pass...");

    VkAttachmentDescription colorAttachment = {};

    colorAttachment.format = swapchainData_.format.format;
    colorAttachment.samples = msaaSamples_;
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
    depthAttachment.samples = msaaSamples_;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve = {};
    colorAttachmentResolve.format = swapchainData_.format.format;
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

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    // The index of the  attachement is the directive from :
    // layout(location = "index") out vec4 color;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkSubpassDependency subPassDep = {};
    subPassDep.srcSubpass = VK_SUBPASS_EXTERNAL; // Refers to the implicit subpass before the render pass (It it was in
                                                 // dstSubpass would be after the render pass)
    subPassDep.dstSubpass = 0;
    subPassDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; // Wait for the swap chain to read the
                                                                             // image
    subPassDep.srcAccessMask = 0;
    // The operation which should wait this subpass are read and write operation on color attachement
    subPassDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    subPassDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &subPassDep;

    VK_CALL(vkCreateRenderPass(logicalDevice_, &renderPassInfo, nullptr, &renderPass_));

    LOG_INFO("Render Pass Created");
}

void Engine::createFramebuffers(VkRenderPass renderPass, const std::vector<VkImageView>& attachements)
{
    LOG_INFO("Creating Framebuffers...");
    swapchainData_.framebuffers.resize(swapchainData_.imageViews.size());
    std::vector<VkImageView> frameAttachements;
    frameAttachements.insert(frameAttachements.begin(), attachements.begin(), attachements.end());

    for(size_t i = 0; i < swapchainData_.imageViews.size(); i++)
    {
        frameAttachements.push_back(swapchainData_.imageViews[i]);
        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(frameAttachements.size());
        framebufferInfo.pAttachments = frameAttachements.data();
        framebufferInfo.layers = 1;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.height = swapchainData_.extent.height;
        framebufferInfo.width = swapchainData_.extent.width;

        VK_CALL(vkCreateFramebuffer(logicalDevice_, &framebufferInfo, nullptr, &swapchainData_.framebuffers[i]));

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

    VK_CALL(vkCreateDescriptorSetLayout(logicalDevice_, &layoutInfo, nullptr, &descriptorSetLayout_));

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

    viewport_ = {};
    viewport_.x = 0;
    viewport_.y = 0;
    viewport_.width = static_cast<float>(swapchainData_.extent.width);
    viewport_.height = static_cast<float>(swapchainData_.extent.height);
    viewport_.minDepth = 0.0f;
    viewport_.maxDepth = 1.0f;

    // The scissor is masking the "out of the scissor rectangle" data from the viewport
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = swapchainData_.extent;

    VkPipelineViewportStateCreateInfo viewPortInfo = {};
    viewPortInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewPortInfo.viewportCount = 1;
    viewPortInfo.pScissors = &scissor;
    viewPortInfo.scissorCount = 1;
    viewPortInfo.pViewports = &viewport_;

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
    multiSampInfo.rasterizationSamples = msaaSamples_;
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
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    VK_CALL(vkCreatePipelineLayout(logicalDevice_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStageInfos;
    pipelineInfo.layout = pipelineLayout_;
    pipelineInfo.pMultisampleState = &multiSampInfo;
    pipelineInfo.pColorBlendState = &colorBlendInfo;
    pipelineInfo.pDepthStencilState = &stencilInfos;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.pInputAssemblyState = &assemblyInfos;
    pipelineInfo.pRasterizationState = &rasterizerInfo;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pViewportState = &viewPortInfo;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;
    // We can make pipeline derivates from other pipelines if they have similarities
    // We have a single pipeline here, so we don't use it
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if(vkCreateGraphicsPipelines(logicalDevice_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_) !=
       VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(logicalDevice_, vertexShaderModule, nullptr);
    vkDestroyShaderModule(logicalDevice_, fragmentShaderModule, nullptr);

    LOG_INFO("Graphics Pipeline Created");
}

void Engine::createCommandPool()
{
    LOG_INFO("Creating Command Pools...");

    // Create a command pool only for graphics operations
    VkCommandPoolCreateInfo commandPoolInfo = {};
    commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolInfo.queueFamilyIndex = indices_.graphicsFamily;

    VK_CALL(vkCreateCommandPool(logicalDevice_, &commandPoolInfo, nullptr, &commandPool_));

    if(indices_.transferAvailable())
    {
        commandPoolInfo.queueFamilyIndex = indices_.transferFamily;

        VK_CALL(vkCreateCommandPool(logicalDevice_, &commandPoolInfo, nullptr, &commandPoolTransfert_));
    }

    LOG_INFO("Command Pools Created");
}

void Engine::createDepthRessources()
{
    VkFormat depthFormat = findDepthFormat();
    utils::createImage(logicalDevice_,
                       swapchainData_.extent.width,
                       swapchainData_.extent.height,
                       1,
                       msaaSamples_,
                       depthFormat,
                       VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       memoryProperties_,
                       depthImage_,
                       depthImageMemory_);

    depthImageView_ = utils::createImageView(logicalDevice_, depthFormat, depthImage_, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    utils::transitionImageLayout(logicalDevice_,
                                 indices_,
                                 getCommandPool(),
                                 getGraphicsQueue(),
                                 depthImage_,
                                 depthFormat,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 1);
}

void Engine::createColorRessources()
{
    VkFormat format = swapchainData_.format.format;

    utils::createImage(logicalDevice_,
                       swapchainData_.extent.width,
                       swapchainData_.extent.height,
                       1,
                       msaaSamples_,
                       format,
                       VK_IMAGE_TILING_OPTIMAL,
                       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                       memoryProperties_,
                       colorImage_,
                       colorMemory_);
    colorImageView_ = utils::createImageView(logicalDevice_, format, colorImage_, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    utils::transitionImageLayout(logicalDevice_,
                                 indices_,
                                 getCommandPool(),
                                 getGraphicsQueue(),
                                 colorImage_,
                                 format,
                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 1);
}

void Engine::recreateCommandBuffer()
{
    vkDeviceWaitIdle(logicalDevice_);
    vkFreeCommandBuffers(
      logicalDevice_, commandPool_, static_cast<uint32_t>(commandBuffers_.size()), commandBuffers_.data());
    createCommandBuffers();
}

void Engine::createVertexBuffer()
{
    LOG_INFO("Creating and Allocating Vertex Buffer");
    VkDeviceSize bufferSize = sizeof(mesh_.vertices[0]) * mesh_.vertices.size();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(
      logicalDevice_, indices_, bufferSize, usage, properties, memoryProperties_, stagingBuffer, stagingBufferMemory);

    void* pData; // Contains a pointer to the mapped memory

    // Documentation : memory must have been created with a memory type that reports VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    //                  flags is reserved for future use of the vulkanAPI.
    vkMapMemory(logicalDevice_, stagingBufferMemory, 0, bufferSize, 0, &pData);
    memcpy(pData, mesh_.vertices.data(), static_cast<std::size_t>(bufferSize));
    vkUnmapMemory(logicalDevice_, stagingBufferMemory);

    utils::createBuffer(logicalDevice_,
                        indices_,
                        bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        memoryProperties_,
                        meshData_.vertexBuffer,
                        meshData_.vertexBufferMemory);
    utils::copyBuffer(
      logicalDevice_, getCommandPoolTransfer(), getTransfertQueue(), stagingBuffer, meshData_.vertexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice_, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice_, stagingBufferMemory, nullptr);
    LOG_INFO("Vertex Buffer Created");
}

void Engine::createVertexIndexBuffer()
{
    LOG_INFO("Creating and Allocating Index Buffer");
    VkDeviceSize bufferSize = sizeof(mesh_.faces[0]) * mesh_.faces.size();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    utils::createBuffer(
      logicalDevice_, indices_, bufferSize, usage, properties, memoryProperties_, stagingBuffer, stagingBufferMemory);

    void* pData; // Contains a pointer to the mapped memory

    // Documentation : memory must have been created with a memory type that reports VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    //                  flags is reserved for future use of the vulkanAPI.
    vkMapMemory(logicalDevice_, stagingBufferMemory, 0, bufferSize, 0, &pData);
    memcpy(pData, mesh_.faces.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(logicalDevice_, stagingBufferMemory);

    utils::createBuffer(logicalDevice_,
                        indices_,
                        bufferSize,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        memoryProperties_,
                        meshData_.indexBuffer,
                        meshData_.indexBufferMemory);
    utils::copyBuffer(
      logicalDevice_, getCommandPoolTransfer(), getTransfertQueue(), stagingBuffer, meshData_.indexBuffer, bufferSize);

    vkDestroyBuffer(logicalDevice_, stagingBuffer, nullptr);
    vkFreeMemory(logicalDevice_, stagingBufferMemory, nullptr);
    LOG_INFO("Index Buffer Created");
}

void Engine::createUniformBuffer()
{
    LOG_INFO("Creating Uniform Buffer...");
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers_.resize(swapchainData_.imageViews.size());
    uniformBuffersMemory_.resize(swapchainData_.imageViews.size());

    for(size_t i = 0; i < swapchainData_.imageViews.size(); i++)
    {
        utils::createBuffer(logicalDevice_,
                            indices_,
                            bufferSize,
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            memoryProperties_,
                            uniformBuffers_[i],
                            uniformBuffersMemory_[i]);
    }

    LOG_INFO("Uniform Buffer Created");
}

void Engine::updateUniformBuffer(uint32_t imageIndex)
{
    uint16_t angle = 0;

    UniformBufferObject ubo = {};

    ubo.model = Matrix4::Identity();

    ubo.view = camera_->getView();

    ubo.projection = camera_->getProjection();

    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = 2 * std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    ubo.lightPos = Vector3(4 * cos(time), 4 * sin(time), 3);

    void* pData;

    for(uint8_t i = 0; i < swapchainData_.imageViews.size(); i++)
    {
        vkMapMemory(logicalDevice_, uniformBuffersMemory_[i], 0, sizeof(UniformBufferObject), 0, &pData);
        memcpy(pData, &ubo, sizeof(UniformBufferObject));
        vkUnmapMemory(logicalDevice_, uniformBuffersMemory_[i]);
    }
}

void Engine::createDescriptorPool()
{
    LOG_INFO("Creating Descriptor Pool...");

    std::array<VkDescriptorPoolSize, 2> descPoolSizes = {};
    // UBO
    descPoolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descPoolSizes[0].descriptorCount = static_cast<uint32_t>(swapchainData_.images.size());

    // Textures
    descPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descPoolSizes[1].descriptorCount = static_cast<uint32_t>(swapchainData_.images.size());

    VkDescriptorPoolCreateInfo descPoolInfo = {};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = static_cast<uint32_t>(descPoolSizes.size());
    descPoolInfo.pPoolSizes = descPoolSizes.data();
    descPoolInfo.maxSets = static_cast<uint32_t>(swapchainData_.images.size());

    VK_CALL(vkCreateDescriptorPool(logicalDevice_, &descPoolInfo, nullptr, &descriptorPool_));

    LOG_INFO("Descriptor Pool Created");
}

void Engine::createDescriptorSets()
{
    LOG_INFO("Creating Descriptor Sets...");
    std::vector<VkDescriptorSetLayout> layouts(swapchainData_.images.size(), descriptorSetLayout_);
    VkDescriptorSetAllocateInfo descAlloc = {};
    descAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAlloc.descriptorPool = descriptorPool_;
    descAlloc.descriptorSetCount = static_cast<uint32_t>(swapchainData_.images.size());
    descAlloc.pSetLayouts = layouts.data();

    descriptorSets_.resize(swapchainData_.images.size());

    VK_CALL(vkAllocateDescriptorSets(logicalDevice_, &descAlloc, descriptorSets_.data()));

    for(size_t i = 0; i < swapchainData_.images.size(); i++)
    {
        VkDescriptorBufferInfo descBufferInfo = {};
        descBufferInfo.buffer = uniformBuffers_[i];
        descBufferInfo.offset = 0;
        descBufferInfo.range = sizeof(UniformBufferObject);

        // VkDescriptorImageInfo imageInfo = {};
        // imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // imageInfo.imageView = lenaTexture_.getImageView();
        // imageInfo.sampler = lenaTexture_.getSampler();

        std::array<VkWriteDescriptorSet, 1> writeInfos = {};
        writeInfos[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeInfos[0].dstSet = descriptorSets_[i];
        writeInfos[0].dstBinding = 0; // binding index in "layout(binding = 0)"
        writeInfos[0].dstArrayElement = 0;
        writeInfos[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writeInfos[0].descriptorCount = 1; // We can update multiple descriptor at once in an array
        writeInfos[0].pBufferInfo = &descBufferInfo;

        // writeInfos[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeInfos[1].dstSet = descriptorSets_[i];
        // writeInfos[1].dstBinding = 1; // binding index in "layout(binding = 0)"
        // writeInfos[1].dstArrayElement = 0;
        // writeInfos[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        // writeInfos[1].descriptorCount = 1; // We can update multiple descriptor at once in an array
        // writeInfos[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(logicalDevice_, static_cast<uint32_t>(writeInfos.size()), writeInfos.data(), 0, nullptr);
    }

    LOG_INFO("Descriptor Sets Created");
}

/*@brief :  Create the command buffers associated with the command pool
 *           We have a command buffer for every image views in the swapchain
 *           to render each one of them
 */
void Engine::createCommandBuffers()
{
    LOG_INFO("Creating and Recording Command Buffers...");
    commandBuffers_.resize(swapchainData_.framebuffers.size());

    VkCommandBufferAllocateInfo allocateBufferInfo = {};
    allocateBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateBufferInfo.commandPool = commandPool_;
    allocateBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; // Can be submitted to queue, and SECONDARY can be
                                                                // called inside a primary command buffer
    allocateBufferInfo.commandBufferCount = (uint32_t)commandBuffers_.size();

    VK_CALL(vkAllocateCommandBuffers(logicalDevice_, &allocateBufferInfo, commandBuffers_.data()));

    // For every command buffer we begin recording it, but we have to specify how
    for(size_t i = 0; i < commandBuffers_.size(); i++)
    {
        VkCommandBufferBeginInfo commandBeginInfo = {};
        commandBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT; // We can resubmit the command buffer
                                                                               // while it's already pending execution
        commandBeginInfo.pInheritanceInfo = nullptr;

        VK_CALL(vkBeginCommandBuffer(commandBuffers_[i], &commandBeginInfo));

        std::array<VkClearValue, 2> clearValues = {};
        clearValues[0].color = {1.0f, (153.0f / 255.0f), (51.0f / 255.0f), 1.0f};
        clearValues[1].depthStencil = {1.0, 0};

        VkRenderPassBeginInfo renderBeginInfo = {};
        renderBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderBeginInfo.pClearValues = clearValues.data();
        renderBeginInfo.renderPass = renderPass_;
        renderBeginInfo.framebuffer = swapchainData_.framebuffers[i];
        renderBeginInfo.renderArea.extent = swapchainData_.extent;
        renderBeginInfo.renderArea.offset = {0, 0};

        vkCmdBeginRenderPass(commandBuffers_[i],
                             &renderBeginInfo,
                             VK_SUBPASS_CONTENTS_INLINE); // Last parameter used to embedd the command for a primary
                                                          // command buffer or secondary
        {
            vkCmdBindPipeline(commandBuffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

            VkBuffer vertexBuffers[] = {meshData_.vertexBuffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffers_[i], 0, 1, vertexBuffers, offsets);

            vkCmdBindIndexBuffer(commandBuffers_[i], meshData_.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(commandBuffers_[i],
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout_,
                                    0,
                                    1,
                                    &descriptorSets_[i],
                                    0,
                                    nullptr);

            // 1 used for the instanced rendering could be higher i think for multiple instanced
            vkCmdDrawIndexed(commandBuffers_[i], static_cast<uint32_t>(mesh_.faces.size() * 3), 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(commandBuffers_[i]);

        VK_CALL(vkEndCommandBuffer(commandBuffers_[i]));
    }

    LOG_INFO("Command Buffers Created");
}

void Engine::createSyncObjects()
{
    LOG_INFO("Creating Synchronization Objects...");
    imageAvailableSemaphore_.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphore_.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        VK_CALL(vkCreateSemaphore(logicalDevice_, &semaphoreInfo, nullptr, &imageAvailableSemaphore_[i]));
        VK_CALL(vkCreateSemaphore(logicalDevice_, &semaphoreInfo, nullptr, &renderFinishedSemaphore_[i]));
        VK_CALL(vkCreateFence(logicalDevice_, &fenceInfo, nullptr, &inFlightFences_[i]));
    }
    LOG_INFO("Synchronization Objects Created");
}

void Engine::checkApplicationState()
{
    for(size_t idx = 0; idx < sizeof(ApplicationStateChange); idx += sizeof(bool))
    {
        bool* state = (reinterpret_cast<bool*>(&applicationChanges_) + idx);

        if(state == &applicationChanges_.modelModified && *state)
        {
            recreateCommandBuffer();
            *state = false;
            return;
        }
        else if(state == &applicationChanges_.modelModified && state)
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

    VK_CALL(vkCreateShaderModule(logicalDevice_, &shaderInfo, nullptr, &shaderModule));

    LOG_INFO("Shader Module Created");

    return shaderModule;
}

VkSampleCountFlagBits Engine::getMaxUsableSampleCount()
{
    VkSampleCountFlags counts = std::min(deviceProperties_.limits.framebufferColorSampleCounts,
                                         deviceProperties_.limits.framebufferDepthSampleCounts);

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
    vkDeviceWaitIdle(logicalDevice_);

    cleanUpSwapChain();
    createSwapChain();
    createRenderPass();
    createGraphicsPipeline();
    createDepthRessources();
    createColorRessources();
    createFramebuffers(renderPass_, {colorImageView_, depthImageView_});
    createCommandBuffers();
}

void Engine::cleanUpSwapChain()
{
    for(auto framebuffer: swapchainData_.framebuffers)
    {
        vkDestroyFramebuffer(logicalDevice_, framebuffer, nullptr);
    }

    vkFreeCommandBuffers(logicalDevice_, commandPool_, commandBuffers_.size(), commandBuffers_.data());

    vkDestroyImageView(logicalDevice_, depthImageView_, nullptr);
    vkDestroyImage(logicalDevice_, depthImage_, nullptr);
    vkFreeMemory(logicalDevice_, depthImageMemory_, nullptr);

    vkDestroyImageView(logicalDevice_, colorImageView_, nullptr);
    vkDestroyImage(logicalDevice_, colorImage_, nullptr);
    vkFreeMemory(logicalDevice_, colorMemory_, nullptr);

    vkDestroyPipeline(logicalDevice_, graphicsPipeline_, nullptr);
    vkDestroyPipelineLayout(logicalDevice_, pipelineLayout_, nullptr);
    vkDestroyRenderPass(logicalDevice_, renderPass_, nullptr);

    for(auto imageView: swapchainData_.imageViews)
    {
        vkDestroyImageView(logicalDevice_, imageView, nullptr);
    }

    vkDestroySwapchainKHR(logicalDevice_, swapchainData_.swapchain, nullptr);
}

void Engine::cleanup()
{
    if(!isCleaned_)
    {
        isCleaned_ = true;
        cleanUpSwapChain();

        // Descriptor Set/Pool
        vkDestroyDescriptorPool(logicalDevice_, descriptorPool_, nullptr);
        vkDestroyDescriptorSetLayout(logicalDevice_, descriptorSetLayout_, nullptr);

        // lenaTexture_.destroy();

        // Vertex/Uniform/Index buffers
        for(size_t i = 0; i < swapchainData_.images.size(); i++)
        {
            vkDestroyBuffer(logicalDevice_, uniformBuffers_[i], nullptr);
            vkFreeMemory(logicalDevice_, uniformBuffersMemory_[i], nullptr);
        }

        // Mesh data
        vkDestroyBuffer(logicalDevice_, meshData_.vertexBuffer, nullptr);
        vkFreeMemory(logicalDevice_, meshData_.vertexBufferMemory, nullptr);
        vkDestroyBuffer(logicalDevice_, meshData_.indexBuffer, nullptr);
        vkFreeMemory(logicalDevice_, meshData_.indexBufferMemory, nullptr);

        // Semaphores
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(logicalDevice_, imageAvailableSemaphore_[i], nullptr);
            vkDestroySemaphore(logicalDevice_, renderFinishedSemaphore_[i], nullptr);
            vkDestroyFence(logicalDevice_, inFlightFences_[i], nullptr);
        }

        // Command Pool
        vkDestroyCommandPool(logicalDevice_, commandPool_, nullptr);

        if(indices_.transferAvailable())
        {
            vkDestroyCommandPool(logicalDevice_, commandPoolTransfert_, nullptr);
        }

        vkDestroyDevice(logicalDevice_, nullptr);
        vkDestroySurfaceKHR(instance_, surface_, nullptr);

        if constexpr(ENABLE_VALIDATION_LAYERS)
        {
            debug::destroyDebugMessenger(instance_, debugMessenger_);
        }

        vkDestroyInstance(instance_, nullptr);
    }
}

/*@brief : Choose the optimal surface format for the swap chain and return it
 */
void Engine::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    if(availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
    {
        swapchainData_.format = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        return;
    }

    for(const auto& format: availableFormats)
    {
        if(format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            swapchainData_.format = format;
            return;
        }
    }

    swapchainData_.format = availableFormats[0];
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
            swapchainData_.presentMode = mode;
            return;
        }
        else if(mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            bestmode = mode;
        }
    }

    swapchainData_.presentMode = bestmode;
}

void Engine::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        swapchainData_.extent = capabilities.currentExtent;
        return;
    }
    else
    {
        VkExtent2D actualExtent = windowExtent_;
        actualExtent.width =
          std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height,
                                       std::min(capabilities.maxImageExtent.height, actualExtent.height));
        swapchainData_.extent = actualExtent;
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
