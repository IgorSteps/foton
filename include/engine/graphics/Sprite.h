#pragma once
#include <engine/gl/glBuffer.h>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include <engine/gl/Shader.h>
class Sprite
{
public:
	Sprite(const std::string& name, float width, float height) : _name(name), _width(width), _height(height) {};
	virtual ~Sprite() {};
	virtual void Init() = 0;
	virtual void Update(float dt) = 0;
	virtual void Draw(std::unique_ptr<Shader>& shader) = 0;
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 1.0f);

protected:
	float _width, _height;
	std::string _name;
	std::unique_ptr<GLBuffer> _buffer;
};