#pragma once
#include <engine/Ray.h>
#include "cuda_runtime.h"
#include <iostream>

class Sphere
{
public:
    Sphere(glm::vec3 center, float radius) 
        :
        _center(center),
        _radius(radius) {}

    __device__ float Hit(const Ray& r) const
    {
        glm::vec3 oc = r.origin - _center;
        float a = glm::dot(r.direction, r.direction);
        float b = 2.0f * glm::dot(oc, r.direction);
        float c = dot(oc, oc) - _radius * _radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) 
        {  
            return -1.0;
        }
        else 
        {
            float t = (-b - sqrt(discriminant)) / (2.0 * a);
            return t;
        }
    }

    __device__ glm::vec3 GetCenter() const
    {
        return _center;
    }

    __device__ float GetRadius() const
    {
        return _radius;
    }

private:
    glm::vec3 _center;
    float _radius;
};