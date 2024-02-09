#pragma once

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <memory>
#include <engine/gl/Shader.h>

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
	void Draw(std::unique_ptr<Shader>& shader);
	void Unbind();

private:
	/// <summary>
	/// Texture ID used by OpenGL.
	/// </summary>
	GLuint _id;
	float _width, _height;
};