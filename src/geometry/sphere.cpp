//#include "geometry/sphere.h"
//
//// Ray-Sphere intersection test.
//__inline__ __host__ __device__ bool Sphere::IsHit(const Ray& r) const
//{
//    glm::vec3 oc = r.Origin() - m_Center;
//
//    float a = glm::dot(r.Direction(), r.Direction());
//    float b = 2.0f * glm::dot(oc, r.Direction());
//    float c = glm::dot(oc, oc) - pow(m_Radius,2);
//    
//    float discriminant = pow(b, 2) - 4 * a * c;
//    
//    return discriminant >= 0;
//}
