#pragma once
#include <glm/glm.hpp>
#include <core/ray.h>
#include "cuda_runtime.h"

class Sphere {
public:
	Sphere(glm::vec3 center, float radius) : m_Center(center), m_Radius(radius) {}
	__host__ __device__ bool IsHit(const Ray& r) const {
        glm::vec3 oc = r.Origin() - m_Center;

        float a = glm::dot(r.Direction(), r.Direction());
        float b = 2.0f * glm::dot(oc, r.Direction());
        float c = glm::dot(oc, oc) - pow(m_Radius, 2);

        float discriminant = pow(b, 2) - 4 * a * c;

        return discriminant >= 0;
	};
private:
	glm::vec3 m_Center;
	float m_Radius;
};