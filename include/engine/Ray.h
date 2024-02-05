#pragma once
#include <glm/glm.hpp>

class Ray
{
public:
	Ray() {}
	Ray(const glm::vec3& origin, const glm::vec3& direction) : _origin(origin), _direction(direction) {}

	glm::vec3 Origin() const;
	glm::vec3 Direction() const;
	glm::vec3 At(float t) const;

private:
	glm::vec3 _origin;
	glm::vec3 _direction;
};