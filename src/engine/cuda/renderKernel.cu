#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>

__global__ 
void renderKernel(glm::vec3* image, Camera* camera, SphereSprite* sphere, int screenWidth, int screenHeight) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= screenWidth || j >= screenHeight) return; // Check bounds

    float u = float(i) / (screenWidth - 1);
    float v = float(j) / (screenHeight - 1);

    Ray ray = camera->GetRay(u, v); // Assuming GetRay is suitable for __device__ execution
    glm::vec3 color = glm::vec3(0, 0, 0); // Default background color

    if (sphere->Intersects(ray)) { // Assuming Intersects is suitable for __device__ execution
        color = glm::vec3(1, 0, 0); // Red color for the sphere
    }

    image[j * screenWidth + i] = color;
}
