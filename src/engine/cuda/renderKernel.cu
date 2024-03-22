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

__device__ bool isInShadow(const Ray& ray, const Sphere* d_spheres, const int numOfSpheres, float lightDist)
{
    HitData tempHit;
    for (int i = 0; i < numOfSpheres; ++i)
    {
        if (d_spheres[i].Hit(ray, 0.001f, lightDist, tempHit))
        {
            if (!d_spheres[i].IsLight())
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    return false;
}

__device__ glm::vec3 ComputePhongIllumination(
    Light* light,
    const HitData& hit,
    const Sphere* d_spheres,
    int numOfSpheres,
    const glm::vec3& objectColor
)
{
    glm::vec3 lightDir = glm::normalize(light->position - hit.point);

    // Setup shadow ray.
    Ray shadowRay;

    shadowRay.origin = hit.point;
    shadowRay.direction = lightDir;
    float distanceToLight = glm::length(light->position - hit.point);

    // Check if the point is in shadow
    bool inShadow = isInShadow(shadowRay, d_spheres, numOfSpheres, distanceToLight);

    // Ambient.
    float ambientStrength = 0.1f;
    glm::vec3 ambient = ambientStrength * light->color;

    if (inShadow) 
    {
        // If in shadow, only ambient light
        return ambient * objectColor;
    }
    else 
    {
        // Diffuse.
        float diff = max(glm::dot(hit.normal, lightDir), 0.0f);
        glm::vec3 diffuse = diff * light->color;

        glm::vec3 result = (diffuse + ambient) * objectColor;
        return result;
    }
}


__global__
void renderKernel(glm::vec3* output, int width, int height, CameraData* camData, Sphere* d_spheres, int numOfSpheres, Light* d_light, Ground* d_ground)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    // Normalise screen coordinates
    float u = float(i) / (width - 1);
    float v = float(j) / (height - 1);

    Ray ray = GetRay(camData, u, v);
    HitData hitData;

    // Keeps track of the closest hit.
    float closestSoFar = INFINITY;
    // Keeps if we've hit anything.
    bool hitSomething = false;
    // Start with black colour.
    glm::vec3 color = glm::vec3(0.0f);

    for (int x = 0; x < numOfSpheres; x++)
    {
        if (d_spheres[x].Hit(ray, 0.001f, closestSoFar, hitData))
        {
            closestSoFar = hitData.t;
            hitSomething = true;

            if (!d_spheres[x].IsLight())
            {
                color += ComputePhongIllumination(d_light, hitData, d_spheres, numOfSpheres, d_spheres[x].GetColour());
            }
            else
            {
                color = d_spheres[x].GetColour();
            }
        }
    }

    if (!hitSomething && d_ground->Hit(ray, closestSoFar, hitData))
    {
        closestSoFar = hitData.t;
        hitSomething = true;
        color = ComputePhongIllumination(d_light, hitData, d_spheres, numOfSpheres, d_ground->groundColor);
    }

    if (!hitSomething)
    {
        glm::vec3 unitDirection = glm::normalize(ray.direction);
        auto a = 0.5f * (unitDirection.y + 1.0f);
        color = (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
    }


    output[j * width + i] = color;
}



// For debugging
// 
__global__ void printDebugSphereProperties(Sphere* spheres, int numSpheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int x = 0; x < numSpheres; ++x)
        {
            // Print properties of the first sphere
            printf("Sphere %i - Center: (%f, %f, %f), Radius: %f\n",
                x,
                spheres[x].GetCenter().x,
                spheres[x].GetCenter().y,
                spheres[x].GetCenter().z,
                spheres[x].GetRadius()
            );
        }
    }
}

void Renderer::RenderUsingCUDA(void* cudaPtr, int numOfSphere)
{
    // Launch CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (1200 + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (800 + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // TODO: Get width/height from engine
    renderKernel << <numBlocks, threadsPerBlock >> > (static_cast<glm::vec3*>(cudaPtr), 1200, 800, d_cameraData, d_spheres, numOfSphere, d_light, d_Ground);
    //printDebugSphereProperties << <1, 1 >> > (d_spheres, numOfSphere);

    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in kernel launch: %s\n", cudaGetErrorString(error));
    }
}