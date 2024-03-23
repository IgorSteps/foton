#pragma once
#include <glad/glad.h>

class PBO {
public:
    PBO(unsigned int width, unsigned int height);
    ~PBO();

    void Bind();
    void Unbind();

    void Update(float width, float height);
    void Draw();

    GLuint GetID() const;

private:
    unsigned int _width, _height;
    GLuint _pbo;
};
