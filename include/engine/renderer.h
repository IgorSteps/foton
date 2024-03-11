#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/cuda/InteropBuffer.h>

#include <engine/cuda/renderData.cuh>
#include "engine/hittables/Sphere.h"
#include <memory>

class Renderer 
{
public:
    Renderer(Camera* camera, std::vector<Sphere>& spheres);
    ~Renderer();

    void UpdateCameraData();
    void Update(std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(void* cudaPtr, int size);

private:
    std::vector<Sphere> _spheres;
    Camera* _camera;
    CameraData* d_cameraData;
    Sphere* d_spheres;
};