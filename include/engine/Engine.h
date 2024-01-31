#pragma once
#include <memory>
#include <iostream>
#include <engine/gl/glBuffer.h>
#include <engine/Window.h>
#include <engine/gl/Shader.h>
#include <engine/graphics/Sprite.h>
#include <glm/glm.hpp>
class Engine {
public:
	Engine();
	void run();
	void init();
	void update(float dt);
	void draw();
	void loadShaders();
private:
	std::unique_ptr<Window> window;
	std::unique_ptr<Sprite> _sprite;
	std::unique_ptr<Shader> _basicShader;
	glm::mat4 _projection;
};