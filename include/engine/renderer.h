#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/cuda/InteropBuffer.h>
#include "engine/hittables/Sphere.h"
#include "engine/Light.h"
#include "engine/grid/Grid.cuh"
#include <memory>

class Renderer 
{
public:
    Renderer(Camera* camera, Light* light, std::vector<Sphere>& spheres, Grid* grid);
    ~Renderer();

    // UpdateCameraData copies new camera data to the device.
    void UpdateCameraData(float width, float height);
    // UpdateLightData copies new light data to the device.
    void UpdateLightData();

    // UpdateSimple updates the PBO from simpel ray tracing without anything. 
    void UpdateSimple(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);
    // UpdatePhong updates the PBO from simple ray tracing with Phong Illumination. 
    void UpdatePhong(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);
    // UpdateGrid updates the PBO from ray tracing using Grid.
    void UpdateGrid(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);

private:
    std::vector<Sphere> h_Spheres;
    Camera* h_Camera;
    Light* h_Light;
    Grid* h_Grid;

    CameraData* d_Camera;
    Sphere* d_Spheres;
    Light* d_Light;
    Grid* d_Grid;

    // CUDA memory management.
    void AllocateDeviceMemory();
    void CopyToDevice();

    // CUDA kernel wrappers:
    void RayTracePhong(float width, float height, void* cudaPtr, int numOfSpheres);
    void RayTraceGrid(float width, float height, void* cudaPtr);
    void RayTraceSimple(float width, float height, void* cudaPtr, int numOfSpheres);
};