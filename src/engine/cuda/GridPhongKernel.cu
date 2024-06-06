#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <engine/Renderer.h>

__device__ glm::vec3 ComputePhongIllumination(
    const Light* light,
    const HitData& hit,
    Grid* d_Grid
)
{
    glm::vec3 lightDir = glm::normalize(light->position - hit.point);

    // Check if the point is in shadow
    Ray shadowRay(hit.point, lightDir);
    float distanceToLight = glm::length(light->position - hit.point);
    HitData temp;
    bool inShadow = d_Grid->Intersect(shadowRay, temp);

    // Ambient.
    float ambientStrength = 0.1f;
    glm::vec3 ambient = ambientStrength * light->color;
    if (inShadow)
    {
        return ambient * hit.colour;
    }
    else
    {
        // Diffuse.
        float diff = max(glm::dot(hit.normal, lightDir), 0.0f);
        glm::vec3 diffuse = diff * light->color;

        glm::vec3 result = (diffuse + ambient) * hit.colour;
        return result;
    }
}

__global__ void PhongGridKernel(glm::vec3* output, int width, int height, CameraData* d_Camera, Grid* d_Grid, Light* d_Light)
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
        //colour = hitData.colour;
        colour = ComputePhongIllumination(d_Light, hitData, d_Grid);
    }
    else
    {
        colour = glm::vec3(1.0f);
    }

    output[j * width + i] = colour;
}

void Renderer::RayTracePhongGrid(float width, float height, void* cudaPtr)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    PhongGridKernel<<<numBlocks, threadsPerBlock>>>(static_cast<glm::vec3*>(cudaPtr), width, height, d_Camera, d_Grid, d_Light);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
