#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <geometry/sphere.h>
#include <glm/glm.hpp>
#include <core/camera.h>
#include <stdio.h>
#include <renderer/renderer.h>

__device__ glm::vec3 CalculateRayColour(const Ray& r, const Sphere& s)
{
    // Colour our sphere red if hit by a ray.
    if (s.IsHit(r))
    {
        return glm::vec3(1, 0, 0);
    }

    glm::vec3 unitDirection = glm::normalize(r.Direction());
    auto a = 0.5f * (unitDirection.y + 1.0f);

    // blendedValue = (1−a)*startValue+a*endValue
    return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
}

__global__ void RenderKernel(glm::vec3* output, int width, int height, Camera camera, Sphere sphere)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    glm::vec3 pixelCenter = camera.UpperLeftPixel() +
        (static_cast<float>(i) * camera.PixelDeltaU()) +
        (static_cast<float>(j) * camera.PixelDeltaV());
    auto rayDirection = pixelCenter - camera.Center();
    Ray r(camera.Center(), rayDirection);

    glm::vec3 pixelColour = CalculateRayColour(r, sphere);
    output[j * width + i] = pixelColour;
}



void Renderer::Render(glm::vec3* output, int width, int height, const Camera& camera, const Sphere& sphere) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    RenderKernel<<<gridSize,blockSize>>>(output, width, height, camera, sphere);
    cudaDeviceSynchronize(); 
}