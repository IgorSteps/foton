#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Camera* camera, Light* light, std::vector<Sphere>& spheres, Grid* grid)
    : h_Camera(camera), h_Light(light), h_Spheres(spheres), h_Grid(grid)
{
    AllocateDeviceMemory();
    CopyToDevice();
}

Renderer::~Renderer()
{
    CUDA_CHECK_ERROR(cudaFree(d_Camera));
    CUDA_CHECK_ERROR(cudaFree(d_Spheres));
    CUDA_CHECK_ERROR(cudaFree(d_Light));
    CUDA_CHECK_ERROR(cudaFree(d_Grid));
}

void Renderer::AllocateDeviceMemory()
{
    CUDA_CHECK_ERROR(cudaMalloc(&d_Camera, sizeof(CameraData)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_Light, sizeof(Light)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_Spheres, h_Spheres.size() * sizeof(Sphere)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_Grid, sizeof(Grid)));
}

void Renderer::CopyToDevice()
{
    UpdateCameraData(1200.0f, 800.0f);
    CUDA_CHECK_ERROR(cudaMemcpy(d_Light, h_Light, sizeof(Light), cudaMemcpyHostToDevice););
    CUDA_CHECK_ERROR(cudaMemcpy(d_Spheres, h_Spheres.data(), h_Spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_Grid, h_Grid, sizeof(Grid), cudaMemcpyHostToDevice));
}

// UpdateCameraData copies new camera data to the device.
void Renderer::UpdateCameraData(float width, float height) {
    CameraData hostCameraData;
    hostCameraData.position = h_Camera->GetPosition();
    hostCameraData.front = h_Camera->GetFront();
    hostCameraData.up = h_Camera->GetUp();
    hostCameraData.right = h_Camera->GetRight();
    hostCameraData.fov = h_Camera->GetZoom();
    hostCameraData.aspectRatio = width / height;

    CUDA_CHECK_ERROR(cudaMemcpy(d_Camera, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice));
}

// UpdateLightData copies new light data to the device.
void Renderer::UpdateLightData()
{
    Light newLightData;
    newLightData.position = h_Light->position;
    newLightData.color = h_Light->color;
    newLightData.intensity = h_Light->intensity;

    CUDA_CHECK_ERROR(cudaMemcpy(d_Light, &newLightData, sizeof(Light), cudaMemcpyHostToDevice));
}

// UpdatePhong updates the PBO from simple ray tracing with Phong Illumination. 
void Renderer::UpdatePhong(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    int numSpheres = static_cast<int>(h_Spheres.size());
    
    RayTracePhong(width, height, cudaPtr, numSpheres);

    interopBuffer->UnmapCudaResource();
}

// UpdateGrid updates the PBO from ray tracing using Grid.
void Renderer::UpdateGrid(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    RayTraceGrid(width, height, cudaPtr);

    interopBuffer->UnmapCudaResource();
}

void Renderer::UpdatePhongGrid(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    RayTracePhongGrid(width, height, cudaPtr);

    interopBuffer->UnmapCudaResource();
}

// UpdateSimple updates the PBO from simpel ray tracing without anything. 
void Renderer::UpdateSimple(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    int numSpheres = static_cast<int>(h_Spheres.size());
    RayTraceSimple(width, height, cudaPtr, numSpheres);

    interopBuffer->UnmapCudaResource();
}
