#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <engine/gl/Shader.h>
#include <memory>

class Texture
{
public:
	Texture(float width, float height);
	~Texture();
	GLuint GetID() const;

	void Init();
	void Update();
	void Bind();
	void ActivateAndBind();
	void Draw(std::unique_ptr<Shader>& shader);
	void Unbind();

private:
	/// <summary>
	/// Texture ID used by OpenGL.
	/// </summary>
	GLuint _id;
	float _width, _height;
};