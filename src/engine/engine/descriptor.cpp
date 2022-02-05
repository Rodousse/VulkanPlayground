#include "engine/descriptor.hpp"

namespace engine::descriptor
{
VkVertexInputBindingDescription getVertexBindingDescription()
{
    VkVertexInputBindingDescription bindingDescription = {};

    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate =
      VK_VERTEX_INPUT_RATE_VERTEX; // Also could be VK_VERTEX_INPUT_RATE_INSTANCE for instance rendering

    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 3> getVertexAttributeDescriptions()
{
    static_assert(sizeof(Floating) == 4);
    std::array<VkVertexInputAttributeDescription, 3> attribDescriptions = {};
    attribDescriptions[0].binding = 0;
    attribDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribDescriptions[0].location = 0;
    attribDescriptions[0].offset = offsetof(Vertex, pos);

    attribDescriptions[1].binding = 0;
    attribDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribDescriptions[1].location = 1;
    attribDescriptions[1].offset = offsetof(Vertex, normal);

    attribDescriptions[2].binding = 0;
    attribDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attribDescriptions[2].location = 2;
    attribDescriptions[2].offset = offsetof(Vertex, uv);

    return attribDescriptions;
}
} // namespace engine::descriptor
