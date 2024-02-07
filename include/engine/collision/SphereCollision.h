#pragma once
#include <glm/glm.hpp>
#include <engine/Ray.h>

class SphereCollision
{
public:
	SphereCollision(glm::vec3 origin, float radius) : _origin(origin), _radius(radius) {};
	virtual ~SphereCollision() {};
	bool Intersects(const Ray& ray);

private:
	glm::vec3 _origin;
	float _radius;
};