#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Camera* camera, std::vector<Sphere>& spheres)
    : _camera(camera), _spheres(spheres)
{
    // Allocate memory for camera and sphere data on the device
    cudaError_t error = cudaMalloc(&d_cameraData, sizeof(CameraData));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for CameraData: %s\n", cudaGetErrorString(error));
    }

    error = cudaMalloc(&d_spheres, _spheres.size() * sizeof(Sphere));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for Spheres: %s\n", cudaGetErrorString(error));
    }
    error = cudaMemcpy(d_spheres, _spheres.data(), _spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy Spheres: %s\n", cudaGetErrorString(error));
    }

    UpdateCameraData();
}

Renderer::~Renderer() {
    cudaFree(d_cameraData);
    cudaFree(d_spheres);
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

void Renderer::Render(std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    int numSpheres = static_cast<int>(_spheres.size());
    // Update the PBO data via cudaPtr.
   RenderUsingCUDA(cudaPtr, numSpheres);

   interopBuffer->UnmapCudaResource();
}
