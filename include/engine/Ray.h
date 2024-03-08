#pragma once
#include <glm/glm.hpp>
#include "cuda_runtime.h"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __device__ glm::vec3 At(float t) const
    {
        return origin + (t * direction);
    }
};