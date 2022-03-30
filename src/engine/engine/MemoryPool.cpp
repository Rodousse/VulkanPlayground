#include "engine/MemoryPool.hpp"

#include "engine/assert.hpp"
#include "engine/typeDefinition.hpp"
#include "engine/utils.hpp"

#include <algorithm>
#include <cassert>
#include <vulkan/vulkan_core.h>

namespace
{
}
namespace engine
{
MemoryOperationResult MemoryPool::allocatePool(VkDevice device,
                                               VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VkMemoryPropertyFlags requiredMemoryProperties,
                                               const VkPhysicalDeviceMemoryProperties& availableDeviceMemoryProperties,
                                               const std::vector<uint32_t>& queueFamilyIndices)
{
    VkBufferCreateInfo bufferInfo = {};

    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.queueFamilyIndexCount = queueFamilyIndices.size();
    bufferInfo.pQueueFamilyIndices = queueFamilyIndices.data();

    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if(queueFamilyIndices.size() > 1)
    {
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
    }
    // Create buffer
    if(const auto result = vkCreateBuffer(device, &bufferInfo, nullptr, &m_buffer); result != VK_SUCCESS)
    {
        if(result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)
        {
            return static_cast<MemoryOperationResult>(result);
        }
        return MemoryOperationResult::UnhandledResult;
    }

    // Allocate the memory on VRAM
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, m_buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
      utils::findMemoryType(memRequirements.memoryTypeBits, requiredMemoryProperties, availableDeviceMemoryProperties);

    if(const auto result = vkAllocateMemory(device, &allocInfo, nullptr, &m_deviceMemory); result != VK_SUCCESS)
    {
        if(result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)
        {
            return static_cast<MemoryOperationResult>(result);
        }
        return MemoryOperationResult::UnhandledResult;
    }

    MemoryPoolResourceData bufferMemoryProperties{};
    bufferMemoryProperties.size = size;
    // Bind the memory allocated with the vertex buffer
    if(const auto result = vkBindBufferMemory(device, m_buffer, m_deviceMemory, 0); result != VK_SUCCESS)
    {
        if(result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY)
        {
            return static_cast<MemoryOperationResult>(result);
        }
        return MemoryOperationResult::UnhandledResult;
    }
    m_size = size;

    if(requiredMemoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        vkMapMemory(device, m_deviceMemory, 0, VK_WHOLE_SIZE, 0, &m_mappedMemoryPtr);
    }
    return MemoryOperationResult::Success;
}
MemoryPoolResource MemoryPool::createResource(VkDeviceSize size)
{
    MemoryPoolResource resource{PRISMO_NULL_HANDLE};
    if(const auto resourceOffset = findSlotAvailableForSize(size, m_size); resourceOffset)
    {
        resource = &(*m_resources.insert({resourceOffset.value(), size}).first);
    }
    return resource;
}
void* MemoryPool::pointerToResource(MemoryPoolResource resource)
{
    assert(m_resources.count(*resource));
    assert(m_mappedMemoryPtr);
    return static_cast<char*>(m_mappedMemoryPtr) + resource->offset;
}

VkBuffer MemoryPool::buffer() const
{
    return m_buffer;
}

void MemoryPool::destroyResource(MemoryPoolResource resource)
{
    m_resources.erase(*resource);
}

void MemoryPool::deallocatePool(VkDevice device)
{
    if(m_mappedMemoryPtr)
    {
        vkUnmapMemory(device, m_deviceMemory);
    }
    vkDestroyBuffer(device, m_buffer, nullptr);
    vkFreeMemory(device, m_deviceMemory, nullptr);
}

// std::pair<MemoryPool::MemoryPoolResourceDataContainer::iterator, VkDeviceSize> MemoryPool::findSlotAvailableForSize(
//   MemoryPoolResourceDataContainer& data, VkDeviceSize resourceSize, VkDeviceSize memoryPoolSize)
//{
//     std::vector<MemoryPoolResource> sortedResourceData;
//     sortedResourceData.reserve(data.size());
//     for(const auto& resource: data)
//     {
//         sortedResourceData.emplace_back(&resource);
//     }
//     std::sort(sortedResourceData.begin(),
//               sortedResourceData.end(),
//               [](const auto* lhs, const auto* rhs) { return lhs->offset < rhs->offset; });
//     VkDeviceSize potentialOffset = 0;
//     for(const auto* resourceData: sortedResourceData)
//     {
//         if(resourceData->offset - potentialOffset >= resourceSize)
//         {
//             return potentialOffset;
//         }
//         potentialOffset = resourceData->offset + resourceData->size;
//     }
//     if(memoryPoolSize - potentialOffset >= resourceSize)
//     {
//         return potentialOffset;
//     }
//     return std::nullopt;
// }
// std::optional<std::pair<MemoryPool::MemoryPoolResourceDataContainer::iterator, VkDeviceSize>>
// MemoryPool::findSlotAvailableForSize(VkDeviceSize resourceSize, VkDeviceSize memoryPoolSize)
std::optional<VkDeviceSize> MemoryPool::findSlotAvailableForSize(VkDeviceSize resourceSize, VkDeviceSize memoryPoolSize)
{
    VkDeviceSize potentialOffset = 0;
    auto it = std::find_if(m_resources.begin(),
                           m_resources.end(),
                           [&potentialOffset, resourceSize](const auto& resourceData)
                           {
                               if(resourceData.offset - potentialOffset >= resourceSize)
                               {
                                   return true;
                               }
                               potentialOffset = resourceData.offset + resourceData.size;
                               return false;
                           });
    if(memoryPoolSize - potentialOffset >= resourceSize)
    {
        return potentialOffset;
    }
    return std::nullopt;
}
} // namespace engine
