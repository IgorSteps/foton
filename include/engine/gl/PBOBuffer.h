#pragma once
#include <glad/glad.h>
class PBO {
public:
    PBO(unsigned int width, unsigned int height)
        : _width(width), _height(height), _pbo(0) {
        glGenBuffers(1, &_pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    ~PBO() {
        if (_pbo) {
            glDeleteBuffers(1, &_pbo);
        }
    }

    void bind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
    }

    void unbind() {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    GLuint getID() const {
        return _pbo;
    }

private:
    unsigned int _width, _height;
    GLuint _pbo;
};
