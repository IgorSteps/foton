#pragma once
#include <memory>
#include <iostream>
#include <renderer/glBuffer.h>
#include <renderer/Window.h>

class Engine {
public:
	Engine();
	void run();
	void init();
	void update(float dt);
	void draw();

private:
	std::unique_ptr<Window> window;
	std::unique_ptr<GLBuffer> buffer;
};