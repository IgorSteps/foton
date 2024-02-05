#include <engine/Ray.h>

glm::vec3 Ray::Origin() const
{
	return _origin;
}

glm::vec3 Ray::Direction() const
{
	return _direction;
}

glm::vec3 Ray::At(float t) const
{
	return _origin + (t * _direction);
}
