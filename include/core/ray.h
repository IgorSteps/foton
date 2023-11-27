#pragma once
#include <glm/glm.hpp>
#include "cuda_runtime.h"

class Ray
{
public:
	Ray(){}
	__host__ __device__
	Ray(const glm::vec3& origin, const glm::vec3& direction) : m_Origin(origin), m_Direction(direction) {}
	__host__ __device__ glm::vec3 Ray::Origin() const
	{
		return m_Origin;
	}

	__host__ __device__ glm::vec3 Ray::Direction() const
	{
		return m_Direction;
	}

	__host__ __device__ glm::vec3 Ray::At(float t) const
	{
		return m_Origin + (t * m_Direction);
	}

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
};