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

    void Update(float dt)
    {
    //    elapsedTime += dt;

    //    // Parameters for the circular motion
    //    float radius = 2.0f; // Radius of the circle
    //    float speed = 0.3f; // Speed of the light movement

    //    // Calculate the new position
    //    float lightX = cos(elapsedTime * speed) * radius;
    //    float lightZ = sin(elapsedTime * speed) * radius;

    //    position = glm::vec3(lightX, position.y, lightZ);
    }
};