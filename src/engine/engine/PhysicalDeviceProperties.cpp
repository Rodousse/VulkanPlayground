#include "engine/PhysicalDeviceProperties.hpp"

#include <set>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan_core.h>

namespace engine
{
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueProperties.data());

    uint32_t index = 0;

    for(const auto& queueProperty: queueProperties)
    {
        if(queueProperty.queueCount > 0 && queueProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily = index;
        }

        if(queueProperty.queueCount > 0 && queueProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            indices.transferFamily = index;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport);

        if(queueProperty.queueCount > 0 && presentSupport == VK_TRUE)
        {
            indices.presentingFamily = index;
        }

        if(indices.isComplete() && indices.transferAvailable())
        {
            break;
        }

        index++;
    }
    return indices;
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.surfaceCapabilities);

    uint32_t surfaceFormatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &surfaceFormatCount, nullptr);

    if(surfaceFormatCount != 0)
    {
        details.surfaceFormats.resize(surfaceFormatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &surfaceFormatCount, details.surfaceFormats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

    if(presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& extensions)
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensionsProperties(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensionsProperties.data());

    std::set<std::string> requiredExtensions(extensions.begin(), extensions.end());

    for(const auto& extension: extensionsProperties)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

VkFormat findSupportedTilingFormat(VkPhysicalDevice device,
                                   const std::vector<VkFormat>& candidates,
                                   VkImageTiling tiling,
                                   VkFormatFeatureFlags features)
{
    for(VkFormat format: candidates)
    {
        VkFormatProperties formatProp;
        vkGetPhysicalDeviceFormatProperties(device, format, &formatProp);

        if(tiling == VK_IMAGE_TILING_LINEAR && (formatProp.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if(tiling == VK_IMAGE_TILING_OPTIMAL && (formatProp.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find a supported format!");
}

bool isDeviceContainingFeatures(const VkPhysicalDeviceFeatures& deviceFeatures,
                                const VkPhysicalDeviceFeatures& requiredFeatures)
{
    const auto* deviceFeaturePtr = reinterpret_cast<const VkBool32*>(&deviceFeatures);
    const auto* requiredFeaturePtr = reinterpret_cast<const VkBool32*>(&requiredFeatures);
    for(size_t featureIndex = 0; featureIndex < sizeof(const VkPhysicalDeviceFeatures) / sizeof(VkBool32);
        featureIndex++)
    {
        const VkBool32* currentFeature = deviceFeaturePtr + featureIndex;
        const VkBool32* currentRequiredFeature = requiredFeaturePtr + featureIndex;

        if(*currentFeature != *currentRequiredFeature && *currentRequiredFeature == VK_TRUE)
        {
            return false;
        }
    }

    return true;
}

} // namespace engine

