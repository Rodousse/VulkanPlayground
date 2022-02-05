#include "engine/Mesh.hpp"

#include <vulkan/vulkan.h>

namespace engine::descriptor
{

VkVertexInputBindingDescription getVertexBindingDescription();

std::array<VkVertexInputAttributeDescription, 3> getVertexAttributeDescriptions();

} // namespace engine::descriptor
