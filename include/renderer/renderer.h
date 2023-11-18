#pragma once
#include <core/image.h>
#include <core/camera.h>
#include <core/ray.h>
class Renderer {
public:
	Renderer(Image& img, Camera& cam): m_Img(img), m_Camera(cam) {}
	void Render(int imageHeight, int imageWidth) ;
	glm::vec3 RayColour(const Ray& r);
private:
	Camera m_Camera;
	Image m_Img;
};