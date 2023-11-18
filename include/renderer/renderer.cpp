#include <renderer/renderer.h>
#include <iostream>
#include <glm/glm.hpp>
#include <utils/colour.h>


void Renderer::Render(int imageHeight, int imageWidth)
{
    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for (int j = 0; j < imageHeight; ++j)
    {
        std::clog << "\rScanlines remaining: " << (imageHeight - j) << ' ' << std::flush;
        for (int i = 0; i < imageWidth; ++i)
        {
            glm::vec3 pixel_center = m_Camera.UpperLeftPixel() + (static_cast<float>(i) * m_Camera.PixelDeltaU()) + (static_cast<float>(j) * m_Camera.PixelDeltaV());
            auto ray_direction = pixel_center - m_Camera.Center();
            Ray r(m_Camera.Center(), ray_direction);

            glm::vec3 pixelColour = RayColour(r);
            WriteColour(std::cout, pixelColour);
        }
    }
    std::clog << "\rDone.                 \n";

}

glm::vec3 Renderer::RayColour(const Ray& r) {
    glm::vec3 unit_direction = glm::normalize(r.Direction());
    auto a = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - a) * glm::vec3(1.0f, 1.0f, 1.0f) + a * glm::vec3(0.5f, 0.7f, 1.0f);
}
