#include <engine/gl/PBOBuffer.h>
#include <glm/glm.hpp>

PBO::PBO(unsigned int width, unsigned int height)
    : _width(width), _height(height), _pbo(0) 
{
    glGenBuffers(1, &_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

PBO::~PBO() 
{
        glDeleteBuffers(1, &_pbo);
}

GLuint PBO::GetID() const {
    return _pbo;
}

void PBO::Bind() 
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
}

void PBO::Unbind()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}