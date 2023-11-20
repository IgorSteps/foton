#include <renderer/renderer.h>
#include <iostream>


int main() 
{
    // Image
    Image image = Image();
    Camera camera = Camera(image.Width(), image.Height(), 1.0f, 2.0f, glm::vec3(0, 0, 0));

    // Sphere
    Sphere sphere(glm::vec3(0.0f, 0.0f, -1.0f), 0.5f);
    
    // Render
    Renderer r(image, camera, sphere);
    r.Render();
}