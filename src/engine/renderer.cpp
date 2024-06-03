#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Camera* camera, Light* light, std::vector<Sphere>& spheres, Grid* grid)
    : h_Camera(camera), h_Light(light), h_Spheres(spheres), h_Grid(grid)
{
    // Allocate memory on the device:
    cudaError_t error = cudaMalloc(&d_Camera, sizeof(CameraData));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for CameraData: %s\n", cudaGetErrorString(error));
    }
    UpdateCameraData(1200.0f, 800.0f);

    error = cudaMalloc(&d_light, sizeof(Light));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for Lights: %s\n", cudaGetErrorString(error));
    }
    UpdateLightData();

    error = cudaMalloc(&d_spheres, h_Spheres.size() * sizeof(Sphere));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for Spheres: %s\n", cudaGetErrorString(error));
    }
    error = cudaMemcpy(d_spheres, h_Spheres.data(), h_Spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy Spheres: %s\n", cudaGetErrorString(error));
    }

    // Grid:
    error = cudaMalloc(&d_Grid, sizeof(Grid));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for Grid: %s\n", cudaGetErrorString(error));
    }
    error = cudaMemcpy(d_Grid, h_Grid, sizeof(Grid), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy Grid: %s\n", cudaGetErrorString(error));
    }
}

Renderer::~Renderer() {
    cudaFree(d_Camera);
    cudaFree(d_spheres);
    cudaFree(d_light);
    cudaFree(d_Grid);
}

void Renderer::UpdateCameraData(float width, float height) {
    CameraData hostCameraData;
    hostCameraData.position = h_Camera->GetPosition();
    hostCameraData.front = h_Camera->GetFront();
    hostCameraData.up = h_Camera->GetUp();
    hostCameraData.right = h_Camera->GetRight();
    hostCameraData.fov = h_Camera->GetZoom();
    hostCameraData.aspectRatio = width / height;

    cudaError_t error = cudaMemcpy(d_Camera, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy new CameraData: %s\n", cudaGetErrorString(error));
    }
}

void Renderer::UpdateLightData()
{
    Light newLightData;
    newLightData.position = h_Light->position;
    newLightData.color = h_Light->color;
    newLightData.intensity = h_Light->intensity;

    cudaError_t error = cudaMemcpy(d_light, &newLightData, sizeof(Light), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy Lights: %s\n", cudaGetErrorString(error));
    }
}

void Renderer::Update(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    int numSpheres = static_cast<int>(h_Spheres.size());
    
    // Update the PBO data via cudaPtr.
    RenderUsingCUDA(width, height, cudaPtr, numSpheres);

    interopBuffer->UnmapCudaResource();
}

void Renderer::UpdateGrid(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);

    // Update the PBO data via cudaPtr.
    RayTraceGrid(width, height, cudaPtr);

    interopBuffer->UnmapCudaResource();
}