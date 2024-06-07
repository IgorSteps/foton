#pragma once
#include "cuda_runtime.h"
#include <glm/glm.hpp>

class Light {
public:
    glm::vec3 position;
    glm::vec3 colour;
    float intensity;
    float elapsedTime = 0.0f;

    Light(const glm::vec3& pos, const glm::vec3& col, float intens)
        : position(pos), colour(col), intensity(intens) {}
    Light() {};

    void Update(float dt)
    {
        elapsedTime += dt;
        // Move the light in a circle areound x and z axis.
        float radius = 2.0f; 
        float speed = 0.3f; 
        float lightX = cos(elapsedTime * speed) * radius;
        float lightZ = sin(elapsedTime * speed) * radius;

        position = glm::vec3(lightX, position.y, lightZ);
    }
};