#pragma once
#include <glm/glm.hpp>

class Ray
{
public:
	Ray(){}
	Ray(const glm::vec3& origin, const glm::vec3& direction) : m_Origin(origin), m_Direction(direction) {}

	glm::vec3 Origin() const;
	glm::vec3 Direction() const;
	glm::vec3 At(float t) const;

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
};