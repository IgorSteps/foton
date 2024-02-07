#include <engine/graphics/Texture.h>

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

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // Linear filtering for minification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // Linear filtering for magnification
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // Clamp to edge for S coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Clamp to edge for T coordinate

	glBindTexture(GL_TEXTURE_2D, 0);
}

/// <summary>
/// Upload texture to GPU.
/// </summary>
void Texture::Upload(const std::vector<glm::vec3>& data)
{
	glBindTexture(GL_TEXTURE_2D, _id);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB, GL_FLOAT, data.data());
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

void Texture::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint Texture::GetID() const 
{ 
	return _id; 
}
