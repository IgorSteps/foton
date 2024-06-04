#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>


__global__
void SimpleKernel(
    glm::vec3* output,
    int width,
    int height,
    CameraData* camData,
    Sphere* d_spheres,
    int numOfSpheres
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    // Normalise screen coordinates
    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);
    Ray ray = Ray(camData, u, v);
    HitData hitData;

    float closestSoFar = INFINITY;
    bool hitSomething = false;
    glm::vec3 colour = glm::vec3(0.0f);

    for (int x = 0; x < numOfSpheres; x++)
    {
        if (d_spheres[x].Hit(ray, 0.001f, closestSoFar, hitData))
        {
            closestSoFar = hitData.t;
            hitSomething = true;
            colour = d_spheres[x].GetColour();
        }
    }

    if (!hitSomething)
    {
        glm::vec3 unitDirection = glm::normalize(ray.direction);
        auto a = 0.5f * (unitDirection.y + 1.0f);
        colour = (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
    }

    output[j * width + i] = colour;
}

void Renderer::RayTraceSimple(float width, float height, void* cudaPtr, int numOfSpheres)
{
    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    SimpleKernel<<<numBlocks, threadsPerBlock>>>(static_cast<glm::vec3*>(cudaPtr), width, height, d_Camera, d_spheres, numOfSpheres);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in Simple kernel launch: %s\n", cudaGetErrorString(error));
    }
}