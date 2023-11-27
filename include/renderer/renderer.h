#pragma once
#include <core/image.h>
#include <core/camera.h>
#include <geometry/sphere.h>

class Renderer {
public:
	Renderer(Image& img, Camera& cam, Sphere& sphere): m_Img(img), m_Camera(cam), m_Sphere(sphere) {}
	void Render() ;
	void Render(glm::vec3* output, int width, int height, const Camera& c, const Sphere& s);

	//glm::vec3 CalculateRayColour(const Ray& r);
private:
	Camera m_Camera;
	Image m_Img;
	Sphere m_Sphere;
};