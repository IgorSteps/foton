#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>

//@TODO: pass them as params.
const unsigned int screenWidth = 1200;
const unsigned int screenHeight = 800;

class Renderer 
{
public:
	std::vector<glm::vec3> image;

    Renderer(Camera* camera, SphereSprite* sphere);
    void Render();

private:
    Camera* _camera;
    SphereSprite* _sphere;
};