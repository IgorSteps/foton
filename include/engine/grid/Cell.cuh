#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <engine/hittables/Sphere.h>

#define CUDA_CHECK_ERROR(call)                                          \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error " << err << " at " << __FILE__ <<  \
            ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

class Cell {
public:
    __host__ Cell();
    __host__ void Add(int sphereIdx);
    __device__ bool Intersect(Sphere* spheres, int numSpheres, const Ray& ray, float tMin, float tMax, HitData& hit) const;
    __host__ void AllocateDeviceMemory();
    __host__ void CopyToDevice();
    __device__ int GetNumSpheres() const;

private:
    std::vector<int> _h_SphereIndxs;
    int _h_NumSpheres;
    int* _d_SphereIndxs;
};