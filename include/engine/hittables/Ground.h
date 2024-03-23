#pragma once
#include "cuda_runtime.h"
#include <glm/glm.hpp>
#include <engine/Ray.h>
#include <engine/hittables/Hitdata.h>

class Ground {
public:
    glm::vec3 point;  
    glm::vec3 normal;
    glm::vec3 groundColor = glm::vec3(0.8f);

    Ground(const glm::vec3& p, const glm::vec3& n) : point(p), normal(glm::normalize(n)) {}

    __device__ bool Hit(const Ray& ray, float& t, HitData& hit)
    {
        float denom = glm::dot(normal, ray.direction);
        if (abs(denom) > 1e-6) 
        {
            // Ensure we're not parallel
            glm::vec3 p0l0 = point - ray.origin;
            t = glm::dot(p0l0, normal) / denom;

            hit.t = t;
            hit.point = ray.origin + t * ray.direction;
            hit.normal = normal;

            return (t >= 0); // Return true if t is positive
        }

        return false; // Parallel or no intersection
    }
};
