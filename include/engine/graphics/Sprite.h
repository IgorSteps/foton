#pragma once
#include <engine/gl/glBuffer.h>
#include <string>
#include <memory>

class Sprite
{
public:
	Sprite(const std::string& name, float width, float height) : _name(name), _width(width), _height(height) {};
	virtual ~Sprite() { _buffer->~GLBuffer(); };
	virtual void Init() = 0;
	virtual void Update(float dt) = 0;
	virtual void Draw() = 0;
protected:
	float _width, _height;
	std::string _name;
	std::unique_ptr<GLBuffer> _buffer;
};