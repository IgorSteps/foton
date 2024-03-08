#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>



__device__ Ray GetRay(const CameraData* cam, float u, float v) {
    float tanFovHalf = tanf(glm::radians(cam->fov / 2.0f));

    float ndcX = (2.0f * u) - 1.0f;
    float ndcY = 1.0f - (2.0f * v);

    float camX = ndcX * cam->aspectRatio * tanFovHalf;
    float camY = ndcY * tanFovHalf;

    glm::vec3 rayDirection = glm::normalize(cam->front + camX * cam->right - camY * cam->up);
    return Ray{ cam->position, rayDirection };
}

__global__ 
void renderKernel(glm::vec3* output, int width, int height, CameraData* camData, Sphere* d_spheres, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    // Normalise screen coordinates
    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);

    Ray ray = GetRay(camData, u, v);
    for (int x = 0; x < size; x++) 
    {
        if (d_spheres[x].Hit(ray))
        {
            output[j * width + i] = glm::vec3(1.0, 0.0, 0.0); // Red color for the sphere
        }
        else 
        {
            glm::vec3 unitDirection = glm::normalize(ray.direction);
            auto a = 0.5f * (unitDirection.y + 1.0f);
            output[j * width + i] = (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
        }
    }
}

__global__ void printDebugSphereProperties(Sphere* spheres, int numSpheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Print properties of the first sphere
        printf("Sphere 0 - Center: (%f, %f, %f), Radius: %f\n",
            spheres[0]._center.x, spheres[0]._center.y, spheres[0]._center.z,
            spheres[0]._radius);
    }
}

void Renderer::RenderUsingCUDA(void* cudaPtr, int size)
{
    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (1200 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (800 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    renderKernel <<<numBlocks, threadsPerBlock>>> (static_cast<glm::vec3*>(cudaPtr), 1200, 800, d_cameraData, d_spheres, size);
    //printDebugSphereProperties << <1, 1 >> > (d_spheres, size);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kernel launch: %s\n", cudaGetErrorString(error));
    }
}