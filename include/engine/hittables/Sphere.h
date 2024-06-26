#pragma once
#include <engine/Ray.h>
#include "cuda_runtime.h"
#include <iostream>
#include <engine/hittables/Hitdata.h>


class Sphere
{
public:
    Sphere(glm::vec3 center,float radius, glm::vec3 colour, bool isLight)
        :
        _center(center),
        _radius(radius),
        _colour(colour),
        _isLight(isLight)
    {}

    __device__ bool Hit(const Ray& r, float tMin, float tMax, HitData& hit) const
    {
        glm::vec3 oc = r.origin - _center;
        float a = glm::dot(r.direction, r.direction);
        float b = 2.0f * glm::dot(oc, r.direction);
        float c = dot(oc, oc) - _radius * _radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) 
        {
            return false;
        }
        
        float sqrted = sqrt(discriminant);
        float root = (-b - sqrted) / (2.0 * a);
        if (root <= tMin || tMax <= root)
        {
            root = (-b + sqrted) / (2.0 * a);
            if (root <= tMin || tMax <= root)
            {
                return false;
            }
        }

        // Set hit data.
        hit.t = root;
        hit.point = r.At(hit.t);
        hit.normal = glm::normalize((hit.point - _center)/_radius);
        hit.colour = _colour;

        return true;
    }

    __host__ __device__ glm::vec3 GetCenter() const
    {
        return _center;
    }

    __host__ __device__ float GetRadius() const
    {
        return _radius;
    }

    __device__ glm::vec3 GetColour() const
    {
        return _colour;
    }

    __device__ bool IsLight() const 
    {
        return _isLight;
    }

private:
    glm::vec3 _center;
    glm::vec3 _colour;
    float _radius;
    bool _isLight;
};