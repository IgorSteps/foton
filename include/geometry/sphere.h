#pragma once
#include <glm/glm.hpp>
#include <core/ray.h>
class Sphere {
public:
	Sphere(glm::vec3 center, float radius) : m_Center(center), m_Radius(radius) {}

	bool IsHit(const Ray& r) const;
private:
	glm::vec3 m_Center;
	float m_Radius;
};