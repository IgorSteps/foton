#include "engine/graphics/Sprite.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/glm.hpp"
#include <string>

Sprite::Sprite() 
{
    position = glm::vec3(-1.0f, 0.0f, 1.0f);
}

void Sprite::Init()
{
    // top left corner at 0,0.
    std::vector<float>  vertices = {
        // Positions    // TexCoords
        -1.0f,  1.0f,    0.0f, 1.0f,
        -1.0f, -1.0f,    0.0f, 0.0f,
         1.0f, -1.0f,    1.0f, 0.0f,
         1.0f,  1.0f,    1.0f, 1.0f,
    };

    std::vector<unsigned int> indices = {
        0, 1, 2, // First Triangle
        2, 3, 0  // Second Triangle
    };

    // Create a GLBuffer object
    _buffer = std::make_unique<GLBuffer>(GL_FLOAT, GL_TRIANGLES);

    // Define attribute information
    AttributeInfo positionAttrib(0, 2, 0);
    AttributeInfo texCoordAttrib(1, 2, 2);
    _buffer->AddAttributeLocation(positionAttrib);
    _buffer->AddAttributeLocation(texCoordAttrib);

    // Set vertex and element data
    _buffer->SetVertexData(vertices);
    _buffer->SetElementData(indices);

    // Upload data to the GPU
    _buffer->UploadData();
    _buffer->Unbind();
}

void Sprite::Update(float dt)
{
}

void Sprite::Draw(std::unique_ptr<Shader>& shader)
{
    _buffer->Bind(false);
    _buffer->Draw();
}
