#pragma once
#include <memory>
#include <iostream>
#include <engine/gl/glBuffer.h>
#include <engine/Window.h>
#include <engine/gl/Shader.h>
#include <engine/graphics/QuadSprite.h>
#include <engine/graphics/SphereSprite.h>
#include <glm/glm.hpp>
#include <engine/renderer.h>
#include <engine/graphics/Texture.h>
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
	std::unique_ptr<SphereSprite> _sphereSprite;
	std::unique_ptr<Shader> _basicShader;
	glm::mat4 _projection;
	std::unique_ptr<Renderer> _renderer;
	std::unique_ptr<Texture> _texture;
};