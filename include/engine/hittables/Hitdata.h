#pragma once
#include <glm/glm.hpp>

struct HitData {
    glm::vec3 point;
    glm::vec3 normal;
    float t;
    glm::vec3 colour;
};