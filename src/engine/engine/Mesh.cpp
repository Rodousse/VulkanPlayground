#include "engine/Mesh.hpp"

namespace engine
{
Mesh::Mesh(Mesh&& other) noexcept
{
    moveIntoThis(std::move(other));
}

Mesh& Mesh::operator=(Mesh&& other) noexcept
{
    moveIntoThis(std::move(other));
    return *this;
}

void Mesh::refreshBoundingBox()
{
    for(const auto& vertex: vertices)
    {
        aabb.min = aabb.min.cwiseMin(vertex.pos).eval();
        aabb.max = aabb.max.cwiseMax(vertex.pos).eval();
    }
}

void Mesh::moveIntoThis(Mesh&& other) noexcept
{
    vertices = std::move(other.vertices);
    faces = std::move(other.faces);
    name = std::move(other.name);
    aabb = other.aabb;
}

} // namespace engine
