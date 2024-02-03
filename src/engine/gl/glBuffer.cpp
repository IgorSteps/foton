#include <engine/gl/glBuffer.h>
#include <stdexcept>

GLBuffer::GLBuffer(GLenum dataType, GLenum mode)
    : 
    _dataType(dataType),
    _mode(mode),
    _elementSize(0),
    _stride(0)
{
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);
}

GLBuffer::~GLBuffer() 
{
    glDeleteBuffers(1, &_ebo);
    glDeleteBuffers(1, &_vbo);
    glDeleteVertexArrays(1, &_vao);
}

void GLBuffer::BindVAO()
{
    glBindVertexArray(_vao);
}

/// <summary>
/// Bind to the VAO and to the VBO (and optionally EBO).
/// </summary>
void GLBuffer::Bind(bool normalised) 
{
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
 
    if (_hasAttributeLocation)
    {
        for (const AttributeInfo& a : _attributes) {
            glVertexAttribPointer(
                a.location,
                a.size,
                _dataType,
                normalised,
                _stride,
                (void*)(a.offset * getTypeSize(_dataType))
            );
            glEnableVertexAttribArray(a.location);
        }
    }
}

void GLBuffer::Unbind() 
{
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

/// <summary>
/// Add an attribute to the VAO and configure it.
/// </summary>
void GLBuffer::AddAttributeLocation(AttributeInfo& info)
{
    _hasAttributeLocation = true;
    info.offset = _elementSize;
    _attributes.push_back(info);

    _elementSize += info.size;
    _stride = _elementSize * getTypeSize(_dataType);
}

void GLBuffer::PushBackVertexData(const std::vector<float>& data)
{
    _vertexData.insert(_vertexData.end(), data.begin(), data.end());
}

void GLBuffer::ClearVertexData()
{
    _vertexData.clear();
}

void GLBuffer::SetVertexData(const std::vector<float>& data)
{
    ClearVertexData();
    PushBackVertexData(data);
}

void GLBuffer::PushBackElementData(const std::vector<unsigned int>& indices) {
    _elementData.insert(_elementData.end(), indices.begin(), indices.end());
}

void GLBuffer::ClearElementData() {
    _elementData.clear();
}

void GLBuffer::SetElementData(const std::vector<unsigned int>& indices) {
    ClearElementData();
    PushBackElementData(indices);
}

/// <summary>
/// Upload vertex and element data to the GPU.
/// </summary>
void GLBuffer::UploadData() 
{
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);

    glBufferData(GL_ARRAY_BUFFER, _vertexData.size() * sizeof(float), _vertexData.data(), GL_STATIC_DRAW);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _elementData.size() * sizeof(unsigned int), _elementData.data(), GL_STATIC_DRAW);
}

/// <summary>
/// Draw the buffer.
/// </summary>
void GLBuffer::Draw()
{
    glDrawElements(_mode, _elementData.size(), GL_UNSIGNED_INT, 0);
}

/// <summary>
/// Returns the size (in bytes) of the data type.
/// </summary>
GLint GLBuffer::getTypeSize(GLenum dataType) const {
    switch (dataType) 
    {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return sizeof(GLfloat);
        case GL_SHORT:
        case GL_UNSIGNED_SHORT:
            return sizeof(GLshort);
        case GL_BYTE:
        case GL_UNSIGNED_BYTE:
            return sizeof(GLbyte);
        default:
            throw std::runtime_error("Unrecognized data type");
    }
}