#pragma once
#include <glm/glm.hpp>
#include "cuda_runtime.h"
#include <engine/Camera.h>

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __device__ Ray(glm::vec3 o, glm::vec3 d)
    {
        origin = o;
        direction = d;
    }
    __device__ Ray(const CameraData* cam, float u, float v)
    {
        float tanFovHalf = tanf(glm::radians(cam->fov / 2.0f));
        float ndcX = (2.0f * u) - 1.0f;
        float ndcY = 1.0f - (2.0f * v);
        float camX = ndcX * cam->aspectRatio * tanFovHalf;
        float camY = ndcY * tanFovHalf;
        glm::vec3 rayDirection = glm::normalize(cam->front + camX * cam->right - camY * cam->up);
        
        origin = cam->position;
        direction = rayDirection;
    }

    __device__ __host__ glm::vec3 At(float t) const
    {
        return origin + (t * direction);
    }
};