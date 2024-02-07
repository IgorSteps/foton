#pragma once

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>

class Texture
{
public:
	Texture(float width, float height);
	~Texture();
	GLuint GetID() const;

	void Init();
	void Upload(const std::vector<glm::vec3>& data);
	void Bind();
	void ActivateAndBind();
	void Unbind();

private:
	/// <summary>
	/// Texture ID used by OpenGL.
	/// </summary>
	GLuint _id;
	float _width, _height;
};