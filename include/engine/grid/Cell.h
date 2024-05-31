#pragma once
#include <vector>

class Cell {
public:
    void Add(int sphereIdx) 
    {
        _sphereIdxs.push_back(sphereIdx);
    }

    __device__ bool Intersect(const Sphere* spheres, const Ray& ray, float tMin, float tMax, HitData& hit) const
    {
        bool hitAnything = false;
        for (int sphereIndex : _sphereIdxs) 
        {
            if (spheres[sphereIndex].Hit(ray, tMin, tMax, hit)) 
            {
                tMax = hit.t;
                hitAnything = true;
            }
        }
        return hitAnything;
    }
private:
    std::vector<int> _sphereIdxs;
};