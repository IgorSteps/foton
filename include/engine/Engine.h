#pragma once
#include <memory>
#include <iostream>
#include <engine/glBuffer.h>
#include <engine/Window.h>
#include <engine/Shader.h>

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
	std::unique_ptr<GLBuffer> buffer;
	std::unique_ptr<Shader> _basicShader;
};