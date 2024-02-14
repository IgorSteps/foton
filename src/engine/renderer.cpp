#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Camera* camera, SphereSprite* sphere)
    : _camera(camera), _sphere(sphere)
{
    // Allocate memory for camera and sphere data on the device
    cudaError_t error = cudaMalloc(&d_cameraData, sizeof(CameraData));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for CameraData: %s\n", cudaGetErrorString(error));
    }

    error = cudaMalloc(&d_sphereData, sizeof(SphereData));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for SphereData: %s\n", cudaGetErrorString(error));
    }

    UpdateCameraData();
    UpdateSphereData();
}

Renderer::~Renderer() {
    cudaFree(d_cameraData);
    cudaFree(d_sphereData);
}

void Renderer::UpdateCameraData() {
    CameraData hostCameraData;
    hostCameraData.position = _camera->GetPosition();
    hostCameraData.front = _camera->GetFront();
    hostCameraData.up = _camera->GetUp();
    hostCameraData.right = _camera->GetRight();
    hostCameraData.fov = _camera->GetZoom();
    hostCameraData.aspectRatio = 1200.0f / 800.0f;

    cudaError_t error = cudaMemcpy(d_cameraData, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy new CameraData: %s\n", cudaGetErrorString(error));
    }
}

void Renderer::UpdateSphereData() {
    SphereData hostSphereData;
    hostSphereData.position = _sphere->position;
    hostSphereData.radius = _sphere->GetRadius();

    cudaError_t error = cudaMemcpy(d_sphereData, &hostSphereData, sizeof(SphereData), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy new SphereData: %s\n", cudaGetErrorString(error));
    }
}

void Renderer::Render(std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);

    // Update the PBO data via cudaPtr.
   RenderUsingCUDA(cudaPtr);

   interopBuffer->UnmapCudaResource();
}
