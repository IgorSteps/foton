#include <core/ray.h>

glm::vec3 Ray::Origin() const
{
	return m_Origin;
}

glm::vec3 Ray::Direction() const
{
	return m_Direction;
}

glm::vec3 Ray::At(float t) const
{
	return m_Origin + (t * m_Direction);
}