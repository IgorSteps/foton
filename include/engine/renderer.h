#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/cuda/InteropBuffer.h>

#include <engine/cuda/renderData.cuh>
#include "engine/hittables/Sphere.h"
#include "engine/Light.h"
#include <memory>

class Renderer 
{
public:
    Renderer(Camera* camera, Light* light, std::vector<Sphere>& spheres);
    ~Renderer();

    void UpdateCameraData();
    void UpdateLightData();
    void Update(std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(void* cudaPtr, int numOfSphere);

private:
    std::vector<Sphere> h_Spheres;
    Camera* h_Camera;
    Light* h_Light;

    CameraData* d_cameraData;
    Sphere* d_spheres;
    Light* d_light;
};