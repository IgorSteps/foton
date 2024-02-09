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

//void Renderer::CopyImageToDevice() {
//    cudaMemcpy(d_image, image.data(), screenWidth * screenHeight * sizeof(glm::vec3), cudaMemcpyHostToDevice);
//}
//
//void Renderer::CopyImageToHost() {
//    cudaMemcpy(image.data(), d_image, screenWidth * screenHeight * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//}

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

void Renderer::Render()
{
    for (int j = 0; j < screenHeight; ++j) {
        for (int i = 0; i < screenWidth; ++i) {
            // Normalise screen coordinates.
            float u = float(i) / (screenWidth - 1);
            float v = float(j) / (screenHeight - 1);

            Ray ray = _camera->GetRay(u, v);
            glm::vec3 color = glm::vec3(0, 0, 0); // Default background color

            if (_sphere->Intersects(ray)) {
                color = glm::vec3(1, 0, 0); // Red color for the sphere
            }
            image[j * screenWidth + i] = color;
        }
    }
}
