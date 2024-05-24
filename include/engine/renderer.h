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
    Renderer( Camera* camera, Light* light, std::vector<Sphere>& spheres);
    ~Renderer();

    void UpdateCameraData(float width, float height);
    void UpdateLightData();
    void Update(float width, float height, std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(float width, float height, void* cudaPtr, int numOfSphere);

private:
    // host entities:
    std::vector<Sphere> h_Spheres;
    Camera* h_Camera;
    Light* h_Light;


    // device entities:

    CameraData* d_cameraData;
    Sphere* d_spheres;
    Light* d_light;
};