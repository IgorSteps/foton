#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/cuda/InteropBuffer.h>

#include <engine/cuda/renderData.cuh>
#include "engine/hittables/Sphere.h"
#include "engine/Light.h"
#include "engine/hittables/Ground.h"
#include <memory>

class Renderer 
{
public:
    Renderer(Ground& ground, Camera* camera, Light* light, std::vector<Sphere>& spheres);
    ~Renderer();

    void UpdateCameraData();
    void UpdateLightData();
    void Update(std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(void* cudaPtr, int numOfSphere);

private:
    // host entities:
    std::vector<Sphere> h_Spheres;
    Camera* h_Camera;
    Light* h_Light;
    Ground h_Ground;

    // device entities:
    Ground* d_Ground;
    CameraData* d_cameraData;
    Sphere* d_spheres;
    Light* d_light;
};