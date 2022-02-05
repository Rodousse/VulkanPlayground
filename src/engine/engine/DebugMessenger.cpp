#include "engine/DebugMessenger.hpp"

#include "engine/assert.hpp"

#include <cstring>
#include <exception>
#include <vector>

namespace engine::debug
{
VkResult CreateDebugUtilMessengerEXT(VkInstance instance,
                                     VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                     VkAllocationCallbacks* pAllocator,
                                     VkDebugUtilsMessengerEXT* pCallback);

VkResult DestroyDebugUtilMessengerEXT(VkInstance instance,
                                      VkDebugUtilsMessengerEXT messenger,
                                      const VkAllocationCallbacks* pAllocator);

void createDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT& messenger, debugCallbackType callback)
{
    LOG_INFO("Set up debug callback");

    VkDebugUtilsMessengerCreateInfoEXT debugInfo = {};
    debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback = callback;
    debugInfo.pUserData = nullptr;

    VK_CALL(CreateDebugUtilMessengerEXT(instance, &debugInfo, nullptr, &messenger))
}

void destroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger)
{
    VK_CALL(DestroyDebugUtilMessengerEXT(instance, messenger, nullptr));
}

VKAPI_ATTR VkBool32 VKAPI_CALL defaultDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* pCallBackData,
                                                           void* pUserData)
{
    switch(messageSeverity)
    {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: LOG_INFO(pCallBackData->pMessage); break;

        //case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: LOG_INFO(pCallBackData->pMessage); break;

        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: LOG_WARNING(pCallBackData->pMessage); break;

        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: LOG_ERROR(pCallBackData->pMessage); break;
    }

    return VK_FALSE;
}

/* @brief : This will look for the function we need to create a messenger
 */
VkResult CreateDebugUtilMessengerEXT(VkInstance instance,
                                     VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                     VkAllocationCallbacks* pAllocator,
                                     VkDebugUtilsMessengerEXT* pCallback)
{
    auto createMessengerFunct =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if(createMessengerFunct != nullptr)
    {
        return createMessengerFunct(instance, pCreateInfo, pAllocator, pCallback);
    }
    else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

/* @brief : This will look for the function we need to destroy and deallocate
 * the messenger we specified in the parameters
 */
VkResult DestroyDebugUtilMessengerEXT(VkInstance instance,
                                      VkDebugUtilsMessengerEXT messenger,
                                      const VkAllocationCallbacks* pAllocator)
{
    auto destroyMessengerFunct =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if(destroyMessengerFunct != nullptr)
    {
        destroyMessengerFunct(instance, messenger, pAllocator);
        return VK_SUCCESS;
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

bool checkValidationLayerSupport()
{
    uint32_t layerCount{};
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for(auto layerName: VALIDATION_LAYERS)
    {
        bool layerFound = false;

        for(const auto& layerProperties: availableLayers)
        {
            if(strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if(!layerFound)
        {
            return false;
        }
    }

    return true;
}

} // namespace engine::debug
