#include <engine/collision/SphereCollision.h>

bool SphereCollision::Intersects(const Ray& r)
{
	glm::vec3 oc = r.Origin() - _origin;
		
	float a = glm::dot(r.Direction(), r.Direction());
	float b = 2.0f * glm::dot(oc, r.Direction());
	float c = glm::dot(oc, oc) - pow(_radius,2);
		    
	float discriminant = pow(b, 2) - 4 * a * c;
		    
	return discriminant >= 0;
}
