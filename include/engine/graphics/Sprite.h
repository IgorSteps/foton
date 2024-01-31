#pragma once
#include <engine/gl/glBuffer.h>
#include <string>
#include <memory>

class Sprite
{
public:
	Sprite(const std::string& name, float width, float height);
	~Sprite();
	void Init();
	void Update(float dt);
	void Draw();
private:
	float _width, _height;
	std::string _name;
	std::unique_ptr<GLBuffer> _buffer;
};