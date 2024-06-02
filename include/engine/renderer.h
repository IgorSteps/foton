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

    void UpdateCameraData(float width, float height);
    void UpdateLightData();
    void Update(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);
    void UpdateGrid(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(float width, float height, void* cudaPtr, int numOfSphere);
    void RayTraceGrid(float width, float height, void* cudaPtr);

private:
    std::vector<Sphere> h_Spheres;
    Camera* h_Camera;
    Light* h_Light;
    Grid* h_Grid;

    CameraData* d_Camera;
    Sphere* d_spheres;
    Light* d_light;
    Grid* d_Grid;
};