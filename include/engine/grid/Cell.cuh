#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <engine/hittables/Sphere.h>

// Macro to check CUDA calls for errors.
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
    // Add adds a sphere index to this cell.
    __host__ void Add(int sphereIdx);
    // Intersect performs ray-sphere intersection tests for all spheres in this cell.
    __device__ bool Intersect(const Sphere* spheres, const int numSpheres, const Ray& ray, float tMin, float tMax, HitData& hit) const;
    // AllocateDeviceMemory allocates memory on the device for sphere indexes.
    __host__ void AllocateDeviceMemory();
    // CopyToDevice copies sphere indexes to the device.
    __host__ void CopyToDevice();
    __device__ int GetNumSpheres() const;

private:
    // _h_SphereIndxs stores sphere indexes for this cell on the host.
    std::vector<int> _h_SphereIndxs;
    // _h_NumSpheres stores number of spheres for this cell on the host.
    int _h_NumSpheres;
    // _d_SphereIndxs stores sphere indexes for this cell on the device.
    int* _d_SphereIndxs;
};