#pragma once

#include "engine/PhysicalDeviceProperties.hpp"

#include <vulkan/vulkan.h>

namespace engine::utils
{

/**
 * @brief Get the maximum usable sample count for framebuffer color and depth resources
 *
 * @param[in] deviceProperties Physical device properties containing max sample counts for each resource
 *
 * @return The maximum usable sample count flag
 */
VkSampleCountFlagBits getMaxUsableSampleCount(const VkPhysicalDeviceProperties& deviceProperties);

/**
 * @brief Checks whether or not, the given format contains a stencil component
 *
 * @param[in] format Format to test
 *
 * @return True if format contains a stencil component
 */
bool hasStencilComponent(VkFormat format);

/**
 * @brief Find the memory type that matches the given properties depending on the output of
 * VkGetBufferMemoryRequirements
 *
 * @param[in] requiredMemoryTypeBits Required memory type bits in VkMemoryRequirements::memoryTypeBits
 * @param[in] properties Memory property flags (Device, host, visible, cached,...)
 * @param[in] memoryProperties Physical device memory properties
 *
 * @return Memory type index to use (e.g. in VkMemoryAllocateInfo::memoryTypeIndex)
 */
uint32_t findMemoryType(uint32_t requiredMemoryTypeBits,
                        VkMemoryPropertyFlags properties,
                        const VkPhysicalDeviceMemoryProperties& memoryProperties);

/**
 * @brief Create a VkBuffer with associated VkDeviceMemory with the given properties
 *
 * @param[in] device Logical device used
 * @param[in] indices Queue family indices used when manipulating this buffer
 * @param[in] size Allocation size
 * @param[in] usage Buffer usage flags
 * @param[in] properties Memory property flags
 * @param[in] memoryProperties Memory properties of the physical device associated with device
 * @param[out] buffer Buffer view created
 * @param[out] bufferMemory Buffer memory created
 */
void createBuffer(VkDevice device,
                  const QueueFamilyIndices& indices,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  const VkPhysicalDeviceMemoryProperties& memoryProperties,
                  VkBuffer& buffer,
                  VkDeviceMemory& bufferMemory);

/**
 * @brief Create a VkImage with its associated VkDeviceMemory with the given properties
 *
 * @param[in] device Logical device used
 * @param[in] width Image width
 * @param[in] height Image height
 * @param[in] mipLevels Number of mip levels wanted
 * @param[in] numSamples Samples per pixel
 * @param[in] format Image format
 * @param[in] tiling Tiling used
 * @param[in] usage Image usage
 * @param[in] property Wanted memory property flags
 * @param[in] memoryProperties Physical device memory properties
 * @param[out] image Created image
 * @param[out] imageMemory Allocated image memory
 * @param[in] flags Additional image creation flags
 */
void createImage(VkDevice device,
                 uint32_t width,
                 uint32_t height,
                 uint32_t mipLevels,
                 VkSampleCountFlagBits numSamples,
                 VkFormat format,
                 VkImageTiling tiling,
                 VkImageUsageFlags usage,
                 VkMemoryPropertyFlags property,
                 const VkPhysicalDeviceMemoryProperties& memoryProperties,
                 VkImage& image,
                 VkDeviceMemory& imageMemory,
                 VkImageCreateFlags flags = 0);

/**
 * @brief Create the image view associated with the given image
 *
 * @param[in] device Logical device used
 * @param[in] format Image format
 * @param[in] image Image to associate the view with
 * @param[in] aspectFlags Image aspect flags
 * @param[in] mipLevels Number of mip levels wanted
 * @param[in] flags Additional image view create flags
 *
 * @return The created image view
 */
VkImageView createImageView(VkDevice device,
                            VkFormat format,
                            VkImage image,
                            VkImageAspectFlags aspectFlags,
                            uint32_t mipLevels,
                            VkImageViewCreateFlags flags = 0);

/**
 * @brief Allocates and begin a single time command that will be submitted to the given VkCommandPool
 *
 * @param[in] device Logical device used
 * @param[in] commandPool Submitted command pool
 *
 * @return A single time command ready for instructions
 */
VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);

/**
 * @brief Submit a single time command created with beginSingleTimeCommands
 *
 * @param[in] device Logical device used
 * @param[in] commandPool Command pool used in beginSingleTimeCommands
 * @param[in] queue Queue family index used for submission
 * @param[in] commandBuffer Command to submit
 */
void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer);

/**
 * @brief Transition the image from one layout to another
 *
 * @param[in] device Logical device to use
 * @param[in] commandPool Command pool used
 * @param[in] queue Queue family index used
 * @param[in] image Image to transition
 * @param[in] format Image format
 * @param[in] oldLayout Old layout
 * @param[in] newLayout New layout
 * @param[in] mipLevels Number of mip levels of the image
 */
void transitionImageLayout(VkDevice device,
                           VkCommandPool commandPool,
                           VkQueue queue,
                           VkImage image,
                           VkFormat format,
                           VkImageLayout oldLayout,
                           VkImageLayout newLayout,
                           uint32_t mipLevels);

/**
 * @brief Copy a buffers content in another one
 *
 * @param[in] device Logical device used
 * @param[in] commandPool Command pool used
 * @param[in] queue Queue family index used for submission
 * @param[in] srcBuffer Buffer to copy from
 * @param[in] dstBuffer Buffer to copy to
 * @param[in] size Size to copy
 */
void copyBuffer(
  VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

/**
 * @brief Copy the content of a buffer into an image
 *
 * @param[in] device Logical device used
 * @param[in] commandPool Command pool used
 * @param[in] queue Queue family index used for submission
 * @param[in] buffer Buffer to copy from
 * @param[in] image Image to copy to
 * @param[in] width Image width
 * @param[in] height Image height
 */
void copyBufferToImage(VkDevice device,
                       VkCommandPool commandPool,
                       VkQueue queue,
                       VkBuffer buffer,
                       VkImage image,
                       uint32_t width,
                       uint32_t height);

} // namespace engine::utils
