#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/Camera.h>
#include <engine/graphics/SphereSprite.h>

const unsigned int screenWidth = 1200;
const unsigned int screenHeight = 800;


class Renderer {
public:
    Renderer(Camera* camera, SphereSprite* sphere) 
        : _camera(camera), _sphere(sphere) 
    {
        image.resize(screenWidth * screenHeight);
    }

	std::vector<glm::vec3> image;
	void Render()
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
private:
    Camera* _camera;
    SphereSprite* _sphere;

};