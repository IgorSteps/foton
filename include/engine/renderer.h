#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>
#include <engine/cuda/renderData.cuh>
//@TODO: pass them as params.
const unsigned int screenWidth = 1200;
const unsigned int screenHeight = 800;

class Renderer 
{
public:
	std::vector<glm::vec3> image;
    glm::vec3* d_image;

    Renderer(Camera* camera, SphereSprite* sphere);
    ~Renderer();

    void UpdateCameraData();
    void UpdateSphereData();
    void Render();
    void RenderUsingCUDA(void* cudaPtr);
    void CopyImageToDevice();
    void CopyImageToHost();
private:
    Camera* _camera;
    SphereSprite* _sphere;
    CameraData* d_cameraData;
    SphereData* d_sphereData;
};