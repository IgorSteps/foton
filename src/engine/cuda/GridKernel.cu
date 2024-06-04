#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>

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
    glm::vec3 colour = glm::vec3(0.0f);

    // Intersect the grid and get the colour of the sphere:
    if (d_Grid->Intersect(ray, hitData))
    {
        colour = hitData.colour;
    }
    else
    {
        colour = glm::vec3(1.0f);
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
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
