#include <engine/Renderer.h>

Renderer::Renderer(Camera* camera, SphereSprite* sphere)
    : _camera(camera), _sphere(sphere)
{
    image.resize(screenWidth * screenHeight);
}

void Renderer::Render()
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

