#pragma once
#include <engine/Ray.h>

class HitData 
{
public:
    glm::vec3 p;
    glm::vec3 normal;
    double t;
};

class Hittable 
{
public:
    virtual ~Hittable() = default;

    virtual bool Hit(const Ray& r, double ray_tmin, double ray_tmax, HitData& rec) const = 0;
};
