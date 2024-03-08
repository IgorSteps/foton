#pragma once
#include <engine/hittables/Hittable.h>
#include "cuda_runtime.h"

class Sphere
{
public:
    Sphere(glm::vec3 center, float radius) 
        :
        _center(center),
        _radius(radius) {}

    __device__ bool Hit(const Ray& r) const
    {
        glm::vec3 oc = r.origin - _center;
        float a = glm::dot(r.direction, r.direction);
        float b = 2.0f * glm::dot(oc, r.direction);
        float c = dot(oc, oc) - _radius * _radius;
        float discriminant = b * b - 4 * a * c;
        return discriminant >= 0;
    }



    glm::vec3 _center;
    float _radius;
private:
};