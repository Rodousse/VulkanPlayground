#pragma once

#include "engine/MemoryPool.hpp"
#include "engine/utils.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>

namespace engine
{

MemoryOperationResult copyMemoryPoolResourceToMemoryPool(const MemoryPool& srcPool,
                                                         MemoryPoolResource srcResource,
                                                         const MemoryPool& dstPool,
                                                         MemoryPoolResource dstResource);

template<typename SrcResourceIterator, typename DstResourceIterator>
void copyMemoryPoolResourceToMemoryPool(VkDevice device,
                                        VkCommandPool commandPool,
                                        VkQueue queue,
                                        const MemoryPool& srcPool,
                                        SrcResourceIterator srcBegin,
                                        SrcResourceIterator srcEnd,
                                        MemoryPool& dstPool,
                                        DstResourceIterator dstBegin)
{
    std::vector<VkBufferCopy> copies{};
    for(; srcBegin != srcEnd; ++srcBegin, ++dstBegin)
    {
        assert((*srcBegin)->size == (*dstBegin)->size);
        copies.emplace_back(VkBufferCopy{(*srcBegin)->offset, (*dstBegin)->offset, (*srcBegin)->size});
    }
    auto command = utils::beginSingleTimeCommands(device, commandPool);

    vkCmdCopyBuffer(command, srcPool.buffer(), dstPool.buffer(), copies.size(), copies.data());

    utils::endSingleTimeCommands(device, commandPool, queue, command);
}

} // namespace engine
