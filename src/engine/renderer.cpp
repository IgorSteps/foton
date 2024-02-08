#include <engine/Renderer.h>
#include "cuda_runtime.h"

Renderer::Renderer(Camera* camera, SphereSprite* sphere)
    : _camera(camera), _sphere(sphere)
{
    image.resize(screenWidth * screenHeight);

    // Allocate memory on the device for the image buffer
    cudaMalloc(&d_image, screenWidth * screenHeight * sizeof(glm::vec3));

    // Allocate memory for camera and sphere data on the device
    cudaMalloc(&d_cameraData, sizeof(CameraData));
    cudaMalloc(&d_sphereData, sizeof(SphereData));

    //// Copy camera and sphere data from host to device
    //CameraData hostCameraData(_camera);
    //cudaMemcpy(d_cameraData, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);

    //SphereData hostSphereData(_sphere);
    //cudaMemcpy(d_sphereData, &hostSphereData, sizeof(SphereData), cudaMemcpyHostToDevice);
    // Update camera and sphere data on the device
    UpdateCameraData();
    UpdateSphereData();
}

Renderer::~Renderer() {
    cudaFree(d_image);
    cudaFree(d_cameraData);
    cudaFree(d_sphereData);
}

void Renderer::CopyImageToDevice() {
    cudaMemcpy(d_image, image.data(), screenWidth * screenHeight * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

void Renderer::CopyImageToHost() {
    cudaMemcpy(image.data(), d_image, screenWidth * screenHeight * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void Renderer::UpdateCameraData() {
    CameraData hostCameraData;
    hostCameraData.position = _camera->GetPosition();
    hostCameraData.front = _camera->GetFront();
    hostCameraData.up = _camera->GetUp();
    hostCameraData.right = _camera->GetRight();
    hostCameraData.fov = _camera->GetZoom();
    hostCameraData.aspectRatio = 1200.0f / 800.0f;

    cudaMemcpy(d_cameraData, &hostCameraData, sizeof(CameraData), cudaMemcpyHostToDevice);
}

void Renderer::UpdateSphereData() {
    SphereData hostSphereData;
    hostSphereData.position = _sphere->position;
    hostSphereData.radius = _sphere->GetRadius();

    cudaMemcpy(d_sphereData, &hostSphereData, sizeof(SphereData), cudaMemcpyHostToDevice);
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
