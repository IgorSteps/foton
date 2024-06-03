#pragma once
#include "Cell.cuh"
#include "engine/hittables/Sphere.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
class Grid {
public:
    __host__ Grid(std::vector<Sphere>& spheres);
    __host__ ~Grid();

    __device__ bool Intersect(const Ray& ray, HitData& hitData);

private:
    glm::vec3 _gridResolution;
    glm::vec3 _gridSize;
    glm::vec3 _cellSize;
    glm::vec3 _gridMin;
    glm::vec3 _gridMax;
    // _balancingFactor determines the appropriate grid resolution for the given scene,
    // after my experiments, for the scene with 50 spheres
    const int _balancingFactor = 0.1;
    // 1D array that stores 3D coords in the following format:
    // First all x coords, then all y coords and lastly all z coords.
    thrust::host_vector<Cell> _h_Cells;
    thrust::host_vector<Sphere> _h_Spheres;
    thrust::device_vector<Sphere> _d_Spheres;
    Cell* _d_Cells;
    int _totalNumSpheres;

    __host__ void ComputeGridSize();
    __host__ void ComputeGridResolution();
    __host__ void Populate();
    __host__ void CopyCellsToDevice(); 
    __host__ glm::vec3 GetCellCoords(const glm::vec3& worldPos) const;
    __device__ __host__ int GetCellIndex(int x, int y, int z) const;
};