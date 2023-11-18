#include <renderer/renderer.h>
#include <iostream>


int main() {

    Image image = Image();
    Camera camera = Camera(image.Width(), image.Height(), 1.0f, 2.0f, glm::vec3(0, 0, 0));
    
    // Render
    Renderer r(image, camera);
    r.Render(image.Height(), image.Width());
}