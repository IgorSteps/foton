#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <engine/hittables/Sphere.h>

class Cell {
public:
    __host__ Cell();
    __host__ void Add(int sphereIdx);
    __device__ bool Intersect(Sphere* spheres, int numSpheres, const Ray& ray, float tMin, float tMax, HitData& hit) const;
    __host__ void AllocateDeviceMemory();
    __host__ void CopyToDevice();

    int _h_NumSpheres;
    int* _d_NumSpheres;
private:
    std::vector<int> _h_SphereIndxs;
    int* _d_SphereIndxs;
};