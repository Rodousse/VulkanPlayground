#pragma once

#ifndef NDEBUG
#include <array>
#include <vulkan/vulkan.h>

namespace engine::debug
{
static constexpr std::array<const char*, 1> VALIDATION_LAYERS = {"VK_LAYER_KHRONOS_validation"};
/*  @brief : Function callBack, called when a debug message coming from vulkan
 *   @param messageSeverity :
 *       VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
 *       VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT
 *       VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
 *       VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
 *
 *   @param messageType :
 *       VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
 *       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
 *       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
 *
 *   @param pCallBackData :
 *       pMessage : contains the message as a string
 *       pObjects : the objects associated with it
 *       objectCount : The number of objects associated
 *
 *   @param pUserData :
 *       A pointer specified during the setup of the callBack. It could be
 * anything !
 */
VKAPI_ATTR VkBool32 VKAPI_CALL defaultDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* pCallBackData,
                                                           void* pUserData);

using debugCallbackType = VKAPI_ATTR VkBool32 VKAPI_CALL (*)(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                                             VkDebugUtilsMessageTypeFlagsEXT,
                                                             const VkDebugUtilsMessengerCallbackDataEXT*,
                                                             void*);

void createDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT& messenger, debugCallbackType callback);
void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger);

bool checkValidationLayerSupport();

} // namespace engine::debug
#endif
