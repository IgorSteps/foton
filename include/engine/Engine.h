#pragma once
#include <memory>
#include <iostream>
#include <engine/gl/glBuffer.h>
#include <engine/Window.h>
#include <engine/gl/Shader.h>
#include <engine/graphics/QuadSprite.h>
#include <glm/glm.hpp>
#include <engine/renderer.h>
#include <engine/graphics/Texture.h>
#include <engine/gl/PBOBuffer.h>
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
	void updateCameraFromEvent(std::unique_ptr<Camera>& camera, float dt);
private:
	std::unique_ptr<Window> _window;
	std::unique_ptr<Camera> _camera;
	std::unique_ptr<QuadSprite> _quadSprite;
	std::unique_ptr<Shader> _basicShader;
	std::unique_ptr<Renderer> _renderer;
	std::unique_ptr<Texture> _texture;
	std::unique_ptr <PBO> _pbo;
	std::unique_ptr <InteropBuffer> _interopBuffer;
	std::vector<Sphere> _spheres;
	// FPS.
	int _frameCount = 0;
	float _timeSinceLastFPSUpdate = 0.0f;
	float _fpsUpdateInterval = 1.0f; // Update FPS every second
	float _lastFPS = 0.0f; // Store the last calculated FPS value
};

