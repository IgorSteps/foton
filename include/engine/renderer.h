#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>
#include <engine/cuda/InteropBuffer.h>

#include <engine/cuda/renderData.cuh>
#include "engine/hittables/Sphere.h"

//@TODO: pass them as params.
const unsigned int screenWidth = 1200;
const unsigned int screenHeight = 800;

class Renderer 
{
public:
    Renderer(Camera* camera, std::vector<Sphere>& spheres);
    ~Renderer();

    void UpdateCameraData();
    void UpdateSphereData();
    void Render(std::unique_ptr<InteropBuffer>& interopBuffer);
    void RenderUsingCUDA(void* cudaPtr, int size);

private:
    std::vector<Sphere> _spheres;
    Camera* _camera;
    CameraData* d_cameraData;
    Sphere* d_spheres;
};