
#pragma once

#include "engine/Logger.hpp"

#include <stdexcept>
#include <vulkan/vulkan.h>

#define VK_CALL(function) \
    if(auto res = function; res != VK_SUCCESS) \
    { \
        LOG_ERROR("Expected VK_SUCCESS and got: " + std::to_string(res)); \
        throw std::runtime_error("Failed to run function"); \
    }
