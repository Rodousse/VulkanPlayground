#pragma once
#include "EngineSymbols.h"
#include "engine/Scene.hpp"

#include <filesystem>
#include <list>
#include <optional>
#include <string>

namespace engine::IO
{
/**
 * @brief load meshes at given path
 * @param[in] path Absolute path to mesh file
 * @param[out] scene Loaded scene from file
 * @return true if everything went well
 */
ENGINE_API std::optional<Scene> loadScene(const std::filesystem::path& path);

} // namespace engine::IO
