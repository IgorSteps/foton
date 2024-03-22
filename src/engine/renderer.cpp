﻿#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Ground& ground, Camera* camera, Light* light, std::vector<Sphere>& spheres)
    : h_Camera(camera), h_Light(light), h_Spheres(spheres), h_Ground(ground)
{
    // Allocate memory on the device:
    cudaError_t error = cudaMalloc(&d_cameraData, sizeof(CameraData));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for CameraData: %s\n", cudaGetErrorString(error));
    }
    UpdateCameraData();

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


    error = cudaMalloc(&d_Ground, sizeof(Ground));
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for Spheres: %s\n", cudaGetErrorString(error));
    }
    error = cudaMemcpy(d_Ground, &h_Ground, sizeof(Ground), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        fprintf(stderr, "Failed to copy Spheres: %s\n", cudaGetErrorString(error));
    }
}

Renderer::~Renderer() {
    cudaFree(d_cameraData);
    cudaFree(d_spheres);
    cudaFree(d_light);
    cudaFree(d_Ground);
}

void Renderer::UpdateCameraData() {
    CameraData hostCameraData;
    hostCameraData.position = h_Camera->GetPosition();
    hostCameraData.front = h_Camera->GetFront();
    hostCameraData.up = h_Camera->GetUp();
    hostCameraData.right = h_Camera->GetRight();
    hostCameraData.fov = h_Camera->GetZoom();
    // TODO: Get width/height from engine
    hostCameraData.aspectRatio = 1200.0f / 800.0f;

    cudaError_t error = cudaMemcpy(d_cameraData, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
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

void Renderer::Update(std::unique_ptr<InteropBuffer>& interopBuffer)
{
    interopBuffer->MapCudaResource();

    size_t size;
    void* cudaPtr = interopBuffer->GetCudaMappedPtr(&size);
    int numSpheres = static_cast<int>(h_Spheres.size());
    
    // Update the PBO data via cudaPtr.
    RenderUsingCUDA(cudaPtr, numSpheres);

    interopBuffer->UnmapCudaResource();
}
