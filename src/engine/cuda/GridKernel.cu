#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>


//__device__ bool Intersect(const Grid* d_Grid, Cell* d_Cells, Sphere* d_Spheres, const Ray& ray, float tMin, float tMax, HitData& hit)
//{
//    glm::vec3 rayGridOrigin = ray.origin - d_Grid->_gridMin;
//    glm::vec3 originCell = rayGridOrigin / d_Grid->_cellSize;
//    glm::vec3 normalisedRayDir = glm::normalize(ray.direction);
//    glm::vec3 deltaT = glm::vec3(0), t = glm::vec3(0);
//
//    // AABB (Axis-Aligned Bounding Box) intersection test
//    float t0 = tMin, t1 = tMax;
//    for (int i = 0; i < 3; ++i)
//    {
//        float invDir = 1.0f / normalisedRayDir[i];
//        float tNear = (d_Grid->_gridMin[i] - ray.origin[i]) * invDir;
//        float tFar = (d_Grid->_gridMax[i] - ray.origin[i]) * invDir;
//
//        if (tNear > tFar)
//        {
//            // Swap
//            float temp = tNear;
//            tNear = tFar;
//            tFar = temp;
//        }
//
//        t0 = tNear > t0 ? tNear : t0;
//        t1 = tFar < t1 ? tFar : t1;
//
//        if (t0 > t1)
//        {
//            return false;
//        }
//    }
//
//    for (int i = 0; i < 3; ++i)
//    {
//        if (normalisedRayDir[i] > 0) // Positive direction.
//        {
//            t[i] = ((floor(originCell[i]) + 1) * d_Grid->_cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
//        }
//        else // Negative direction.
//        {
//            t[i] = ((ceil(originCell[i]) - 1) * d_Grid->_cellSize[i] - rayGridOrigin[i]) / normalisedRayDir[i];
//        }
//        deltaT[i] = d_Grid->_cellSize[i] / std::abs(normalisedRayDir[i]);
//    }
//
//    float currentT = 0.0f;
//    while (true)
//    {
//        // Check if the ray intersects any spheres in the current cell
//        int cellIdx = d_Grid->GetCellIndex(static_cast<int>(originCell.x), static_cast<int>(originCell.y), static_cast<int>(originCell.z));
//        Cell cell = d_Cells[cellIdx];
//        for (int i = 0; i < numSpheres; ++i)
//        {
//            if (d_Spheres[_d_SphereIndxs[i]].Hit(ray, tMin, tMax, hit))
//            {
//                tMax = hit.t;
//                return tru
//            }
//        }
//
//
//
//        // Determine the next cell to step to:
//        if (t.x < t.y)
//        {
//            currentT = t.x;
//            t.x += deltaT.x;
//            if (normalisedRayDir.x > 0) // Positive direction.
//            {
//                originCell.x += 1;
//            }
//            else // Negatve direction.
//            {
//                originCell.x -= 1;
//            }
//        }
//        else if (t.y < t.z)
//        {
//            currentT = t.y;
//            t.y += deltaT.y;
//            if (normalisedRayDir.y > 0) // Positive direction.
//            {
//                originCell.y += 1;
//            }
//            else // Negatve direction.
//            {
//                originCell.y -= 1;
//            }
//        }
//        else
//        {
//            currentT = t.z;
//            t.z += deltaT.z;
//            if (normalisedRayDir.z > 0) // Positive direction.
//            {
//                originCell.z += 1;
//            }
//            else // Negatve direction.
//            {
//                originCell.z -= 1;
//            }
//        }
//
//        if (originCell.x < 0 || originCell.x > d_Grid->_gridMax.x - 1.0f ||
//            originCell.y < 0 || originCell.y > d_Grid->_gridMax.y - 1.0f ||
//            originCell.z < 0 || originCell.z > d_Grid->_gridMax.z - 1.0f)
//        {
//            break;
//        }
//    }
//
//    return false;
//}

__global__ void GridKernel(glm::vec3* output, int width, int height, CameraData* d_Camera, Grid* d_Grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Check for out of bounds.
    if (i >= width || j >= height) return;

    // Create ray using normalised coords.
    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);
    Ray ray = Ray(d_Camera, u, v);
    HitData hitData;
    float tMin = 0.0001f, tMax = INFINITY;
    glm::vec3 colour = glm::vec3(0.0f);

    // Intersect the grid and get the colour of the sphere:
    if (d_Grid->Intersect(ray, tMin, tMax, hitData))
    {
        colour = hitData.colour;
    }
    else
    {
        colour = glm::vec3(1.0f); // White background.
    }

    output[j * width + i] = colour;
}

void Renderer::RayTraceGrid(float width, float height, void* cudaPtr)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    GridKernel<<<numBlocks, threadsPerBlock>>>(static_cast<glm::vec3*>(cudaPtr), width, height, d_Camera, d_Grid);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Grid kernel : % s\n", cudaGetErrorString(error));
    }
}