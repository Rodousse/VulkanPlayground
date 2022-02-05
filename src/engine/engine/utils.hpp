#pragma once

#include "engine/PhysicalDeviceProperties.hpp"

#include <vulkan/vulkan.h>

namespace engine::utils
{

VkImageView createImageView(VkDevice device,
                            VkFormat format,
                            VkImage image,
                            VkImageAspectFlags aspectFlags,
                            uint32_t mipLevels,
                            VkImageViewCreateFlags flags = 0);

VkSampleCountFlagBits getMaxUsableSampleCount(VkPhysicalDeviceProperties deviceProperties);

bool hasStencilComponent(VkFormat format);

uint32_t findMemoryType(uint32_t typeFilter,
                        VkMemoryPropertyFlags properties,
                        const VkPhysicalDeviceMemoryProperties& memoryProperties);

void createBuffer(VkDevice device,
                  const QueueFamilyIndices& indices,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  const VkPhysicalDeviceMemoryProperties& memoryProperties,
                  VkBuffer& buffer,
                  VkDeviceMemory& bufferMemory);

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

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool commandPool);

void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool, VkQueue queue, VkCommandBuffer commandBuffer);

void transitionImageLayout(VkDevice device,
                           const QueueFamilyIndices& indices,
                           VkCommandPool commandPool,
                           VkQueue queue,
                           VkImage image,
                           VkFormat format,
                           VkImageLayout oldLayout,
                           VkImageLayout newLayout,
                           uint32_t mipLevels);
void copyBuffer(
  VkDevice device, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

void copyBufferToImage(VkDevice device,
                       VkCommandPool commandPool,
                       VkQueue queue,
                       VkBuffer buffer,
                       VkImage image,
                       uint32_t width,
                       uint32_t height);

} // namespace engine::utils
