#pragma once

#include <glm/glm.hpp>
class Transform
{
public:
	glm::vec3 Position = glm::vec3(0.0f);
	float Rotation = 0;
	glm::vec3 Scale = glm::vec3(1.0f);
};