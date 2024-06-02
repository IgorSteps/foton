#include <engine/grid/Cell.cuh>
#define CUDA_CHECK_ERROR(call)                                          \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error " << err << " at " << __FILE__ <<  \
            ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__host__ Cell::Cell() :_h_NumSpheres(0), _d_SphereIndxs(nullptr), _d_NumSpheres(nullptr) {}

__host__ void Cell::Add(int sphereIdx)
{
    _h_SphereIndxs.push_back(sphereIdx);
    _h_NumSpheres = _h_SphereIndxs.size();
}

__device__ bool Cell::Intersect(Sphere* spheres, int numSpheres, const Ray& ray, float tMin, float tMax, HitData& hit) const
{
    //if (numSpheres != 0) printf("Num of spheres: %d\n", numSpheres);
    bool hitAnything = false;
    for (int i = 0; i < numSpheres; ++i)
    {   
        if (spheres[_d_SphereIndxs[i]].Hit(ray, tMin, tMax, hit))
        {
            tMax = hit.t;
            hitAnything = true;
        }
    }

    return hitAnything;
}

__host__ void Cell::CopyToDevice()
{
    CUDA_CHECK_ERROR(cudaMemcpy(_d_SphereIndxs, _h_SphereIndxs.data(), _h_NumSpheres * sizeof(int), cudaMemcpyHostToDevice));
}

__host__ void Cell::AllocateDeviceMemory()
{
    CUDA_CHECK_ERROR(cudaMalloc(&_d_SphereIndxs, _h_NumSpheres * sizeof(int)));
}
