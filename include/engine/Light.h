#pragma once
#include "cuda_runtime.h"
#include <glm/glm.hpp>

class Light {
public:
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
    float elapsedTime = 0.0f;

    Light(const glm::vec3& pos, const glm::vec3& col, float intens)
        : position(pos), color(col), intensity(intens) {}
    Light() {};
};