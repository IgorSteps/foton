#include <engine/grid/Cell.cuh>

__host__ Cell::Cell() :_h_NumSpheres(0), _d_SphereIndxs(nullptr) {}

__device__ int Cell::GetNumSpheres() const
{
    return _h_NumSpheres;
}

__host__ void Cell::Add(int sphereIdx)
{
    _h_SphereIndxs.push_back(sphereIdx);
    _h_NumSpheres = _h_SphereIndxs.size();
}

__device__ bool Cell::Intersect(Sphere* spheres, int numSpheres, const Ray& ray, float tMin, float closestSoFar, HitData& hitData) const
{
    bool hitSphere = false;
    for (int i = 0; i < numSpheres; ++i)
    {
        if (spheres[_d_SphereIndxs[i]].Hit(ray, tMin, closestSoFar, hitData))
        {
            closestSoFar = hitData.t;
            hitSphere = true;
        }
    }

    return hitSphere;
}

__host__ void Cell::CopyToDevice()
{
    CUDA_CHECK_ERROR(cudaMemcpy(_d_SphereIndxs, _h_SphereIndxs.data(), _h_NumSpheres * sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void Cell::AllocateDeviceMemory()
{
    CUDA_CHECK_ERROR(cudaMalloc(&_d_SphereIndxs, _h_NumSpheres * sizeof(int)));
}
