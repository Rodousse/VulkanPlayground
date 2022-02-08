#pragma once
#include "engine/Camera.hpp"

namespace engine
{
class ENGINE_API PerspectiveCamera : public Camera
{
  private:
    float m_fov = M_PI / 2.0f;
    void refreshProjection() override;

  public:
    PerspectiveCamera();
    ~PerspectiveCamera() override = default;

    void setFovDeg(float deg);
    void setFovRad(float rad);
    [[nodiscard]] float fov() const;
};

} // namespace engine
