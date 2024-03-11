#pragma once
#include <memory>
#include <iostream>
#include <glm/glm.hpp>
#include <engine/gl/Shader.h>
#include <engine/Window.h>
#include <engine/renderer.h>
#include <engine/RayTracedImage.h>
#include <engine/cuda/InteropBuffer.h>
#include <engine/hittables/Sphere.h>

class Engine {
public:
	Engine();
	void run();
	void init();
	void update(float dt);
	void draw();
	void loadShaders();

private:
	std::unique_ptr<Window> _window;
	std::unique_ptr<Camera> _camera;
	std::unique_ptr<Shader> _shader;
	std::unique_ptr<Renderer> _renderer;
	std::unique_ptr<RayTracedImage> _rayTracedImage;
	std::unique_ptr <InteropBuffer> _interopBuffer;
	std::vector<Sphere> _spheres;

	// FPS.
	int _frameCount = 0;
	float _timeSinceLastFPSUpdate = 0.0f;
	float _fpsUpdateInterval = 1.0f; // Update FPS every second
	float _lastFPS = 0.0f; // Store the last calculated FPS value
};

