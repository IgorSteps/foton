#pragma once
#include <glad/glad.h>

class PBO {
public:
    PBO(float width, float height);
    ~PBO();

    void Bind();
    void Unbind();

    void Update(float width, float height);

    GLuint GetID() const;

private:
    float _width, _height;
    GLuint _pbo;
};
