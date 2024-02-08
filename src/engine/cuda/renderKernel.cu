#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>

struct SRay {
    glm::vec3 origin;
    glm::vec3 direction;
};

__device__ SRay GetRay(const CameraData* cam, float u, float v) {
    float tanFovHalf = tanf(glm::radians(cam->fov / 2.0f));

    float ndcX = (2.0f * u) - 1.0f;
    float ndcY = 1.0f - (2.0f * v);

    float camX = ndcX * cam->aspectRatio * tanFovHalf;
    float camY = ndcY * tanFovHalf;

    glm::vec3 rayDirection = glm::normalize(cam->front + camX * cam->right - camY * cam->up);
    return SRay{ cam->position, rayDirection };
}

__device__ bool Intersects(const SRay& ray, const SphereData* sphere) {
    glm::vec3 oc = ray.origin - sphere->position;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere->radius * sphere->radius;
    float discriminant = b * b - 4 * a * c;
    return discriminant >= 0;
} 


__global__ 
void renderKernel(glm::vec3* output, int width, int height, CameraData* camData, SphereData* sphereData) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    // Normalise screen coordinates
    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);

    // Assuming GetRay and Intersects are implemented in a device-friendly manner
    SRay ray = GetRay(camData, u, v);
    if (Intersects(ray, sphereData)) {
        output[j * width + i] = glm::vec3(1.0, 0.0, 0.0); // Red color for the sphere
    }
    else {
        output[j * width + i] = glm::vec3(0.0, 0.0, 0.0); // Default background color
    }
}

void Renderer::RenderUsingCUDA()
{
    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (1200 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (800 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    renderKernel << <numBlocks, threadsPerBlock >> > (d_image, 1200, 800, d_cameraData, d_sphereData);

    cudaDeviceSynchronize();
}