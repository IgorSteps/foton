#pragma once
#include <glad/glad.h>
#include <vector>

struct AttributeInfo
{
	GLuint location;
	GLint size;
	GLsizei offset;

	AttributeInfo(GLuint location, GLint size, GLsizei offset)
		: location(location), size(size), offset(offset) {}
};

class GLBuffer 
{
public:
	GLBuffer(GLenum dataType, GLenum mode);
	~GLBuffer();

	// Disallow Move and Copy semantics, because it doesn't make sense for OpenGL buffers 
	// as duplicating OpenGL buffers can lead to errors and undefined behavior.
	GLBuffer(const GLBuffer&) = delete;
	GLBuffer& operator=(const GLBuffer&) = delete;
	GLBuffer(GLBuffer&&) = delete;
	GLBuffer& operator=(GLBuffer&&) = delete;
	void BindVAO();
	void Bind(bool normalised);
	void Unbind();

	void AddAttributeLocation(AttributeInfo& info);

	// For VBOs.
	void PushBackVertexData(const std::vector<float>& data);
	void ClearVertexData();
	void SetVertexData(const std::vector<float>& data);
	void UploadData();

	// For EBOs.
	void PushBackElementData(const std::vector<unsigned int>& indices);
	void ClearElementData();
	void SetElementData(const std::vector<unsigned int>& indices);

	void Draw();

private:
	GLuint _vao, _vbo, _ebo;
	GLenum _dataType, _mode;
	GLsizei _elementSize, _stride;
	bool _hasAttributeLocation = false;
	std::vector<float> _vertexData;
	std::vector<unsigned int> _elementData;
	std::vector<AttributeInfo> _attributes;

	GLint getTypeSize(GLenum dataType) const;
};