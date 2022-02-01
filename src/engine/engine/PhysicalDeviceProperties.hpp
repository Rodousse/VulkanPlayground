#pragma once

#include <vector>
#include <vulkan/vulkan.h>

namespace engine
{
struct QueueFamilyIndices
{
    int graphicsFamily = -1;
    int presentingFamily = -1;
    int transferFamily = -1;

    [[nodiscard]] inline bool isComplete() const
    {
        return graphicsFamily >= 0 && presentingFamily >= 0;
    }

    [[nodiscard]] inline bool transferAvailable() const
    {
        return transferFamily >= 0 && transferFamily != graphicsFamily;
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    std::vector<VkSurfaceFormatKHR> surfaceFormats;
    std::vector<VkPresentModeKHR> presentModes;
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& extensions);
VkFormat findSupportedTilingFormat(VkPhysicalDevice device,
                                   const std::vector<VkFormat>& candidates,
                                   VkImageTiling tiling,
                                   VkFormatFeatureFlags features);
bool isDeviceContainingFeatures(const VkPhysicalDeviceFeatures& deviceFeatures,
                                const VkPhysicalDeviceFeatures& requiredFeatures);
} // namespace engine
