#include <engine/graphics/Texture.h>

#include <iostream>
Texture::Texture(float width, float height) : _width(width), _height(height), _id(0)
{
	glGenTextures(1, &_id);
}

Texture::~Texture()
{
	glDeleteTextures(1, &_id);
}

/// <summary>
/// Configure texture.
/// </summary>
void Texture::Init()
{
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB, GL_FLOAT, nullptr);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::Update(float width, float height)
{
	ActivateAndBind();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_FLOAT, nullptr);
	Unbind();
}

void Texture::Bind()
{
	glBindTexture(GL_TEXTURE_2D, _id);
}

void Texture::ActivateAndBind()
{
	glActiveTexture(GL_TEXTURE0);
	Bind();
}

void Texture::Draw(std::unique_ptr<Shader>& shader)
{
	ActivateAndBind();
	auto textLocation = shader->GetUniformLocation("u_texture");
	glUniform1i(textLocation, 0);
}

void Texture::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint Texture::GetID() const 
{ 
	return _id; 
}
