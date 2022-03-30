#pragma once

#include <optional>
#include <set>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace engine
{

enum class MemoryOperationResult
{
    Success = VK_SUCCESS,
    OutOfHostMemory = VK_ERROR_OUT_OF_HOST_MEMORY,
    OutOfDeviceMemory = VK_ERROR_OUT_OF_DEVICE_MEMORY,
    MemoryMapFailed = VK_ERROR_MEMORY_MAP_FAILED,
    UnhandledResult
};

struct MemoryPoolResourceData
{
    VkDeviceSize offset;
    VkDeviceSize size;
};

using MemoryPoolResource = const MemoryPoolResourceData*;

class MemoryPool
{
  public:
    MemoryPool() = default;
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryPool(MemoryPool&&) noexcept = default;
    MemoryPool& operator=(MemoryPool&&) noexcept = default;
    ~MemoryPool() = default;

    MemoryOperationResult allocatePool(VkDevice device,
                                       VkDeviceSize size,
                                       VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags requiredMemoryProperties,
                                       const VkPhysicalDeviceMemoryProperties& availableDeviceMemoryProperties,
                                       const std::vector<uint32_t>& queueFamilyIndices);

    [[nodiscard]] MemoryPoolResource createResource(VkDeviceSize size);

    /**
     * @brief Sets a pointer to the physical location of the resource.
     * Resource must be a valid resource resulting from a previous call from createResource.
     * Pointer shall not be used to access data beyond
     * the size of the resource.
     * @param[in] resource Resource to map
     * @warning The memory pool must be allocated with an host visibility flag
     * @return Pointer to the resource adress
     */
    [[nodiscard]] void* pointerToResource(MemoryPoolResource resource);
    [[nodiscard]] VkBuffer buffer() const;

    void destroyResource(MemoryPoolResource resource);

    void deallocatePool(VkDevice device);

  private:
    struct MemoryPoolResourceDataCompare
    {
        constexpr bool operator()(const MemoryPoolResourceData& lhs, const MemoryPoolResourceData& rhs) const
        {
            return lhs.offset < rhs.offset;
        }
    };
    using MemoryPoolResourceDataContainer = std::set<MemoryPoolResourceData, MemoryPoolResourceDataCompare>;

    std::optional<VkDeviceSize> findSlotAvailableForSize(VkDeviceSize resourceSize, VkDeviceSize memoryPoolSize);
    MemoryPoolResourceDataContainer m_resources;
    VkDeviceSize m_size;
    VkBuffer m_buffer;
    VkDeviceMemory m_deviceMemory;
    void* m_mappedMemoryPtr{nullptr};
};
} // namespace engine
