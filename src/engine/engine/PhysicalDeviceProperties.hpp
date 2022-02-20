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

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    std::vector<VkSurfaceFormatKHR> surfaceFormats;
    std::vector<VkPresentModeKHR> presentModes;
};

/**
 * @brief Find queue families available on the device and pick one from each
 *
 * @param[in] device Physical device to retrieve from
 * @param[in] surface Surface to retrieve present queue from
 *
 * @return A QueueFamilyIndices object
 */
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

/**
 * @brief Query the swapchain support details for the given surface
 *
 * @param[in] device Physical device to retrieve from
 * @param[in] surface Surface to retrieve from
 *
 * @return A SwapchainSupportDetails object
 */
SwapchainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

/**
 * @brief Check if a list of extensions are supported by the given device
 *
 * @param[in] device Physical device to retrieve from
 * @param[in] extensions List of desired extension
 *
 * @return True if all extensions are supported by the device
 */
bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& extensions);

/**
 * @brief Find a format supported with the given image tiling and format features
 *
 * @param[in] device Physical device to retrieve from
 * @param[in] candidates Candidate formats
 * @param[in] tiling Desired tiling
 * @param[in] features Desired format features
 *
 * @return The first supported format from the candidates
 */
VkFormat findSupportedTilingFormat(VkPhysicalDevice device,
                                   const std::vector<VkFormat>& candidates,
                                   VkImageTiling tiling,
                                   VkFormatFeatureFlags features);

/**
 * @brief Check that a device contains all the given features
 *
 * @param[in] deviceFeatures Features providen by the physical device
 * @param[in] requiredFeatures Features desired
 *
 * @return True if all features marked true in requiredFeatures are contained in the deviceFeatures
 */
bool isDeviceContainingFeatures(const VkPhysicalDeviceFeatures& deviceFeatures,
                                const VkPhysicalDeviceFeatures& requiredFeatures);
} // namespace engine
