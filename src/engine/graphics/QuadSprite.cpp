#include "engine/graphics/QuadSprite.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext.hpp"
QuadSprite::QuadSprite(const std::string& name, float width, float height) 
    : Sprite(name, width, height)
{
    position = glm::vec3(-1.0f, 0.0f, 1.0f);
}

void QuadSprite::Init()
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
    AttributeInfo positionAttrib(0, 2, 0); // Assuming location 0 for position
    AttributeInfo texCoordAttrib(1, 2, 2); // Assuming location 1 for texture coords, with an offset of 2 floats
    _buffer->AddAttributeLocation(positionAttrib);
    _buffer->AddAttributeLocation(texCoordAttrib);

    // Set vertex and element data
    _buffer->SetVertexData(vertices);
    _buffer->SetElementData(indices);

    // Upload data to the GPU
    _buffer->UploadData();
    _buffer->Unbind();
}

void QuadSprite::Update(float dt)
{
}

void QuadSprite::Draw(std::unique_ptr<Shader>& shader)
{
    // colour
    /*auto colorLocation = shader->GetUniformLocation("myColor");
    glUniform4f(colorLocation, 0.0f, 1.0f, 0.0f, 1.0f);*/

    // model
    /*glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, position);
    auto modelLocation = shader->GetUniformLocation("u_model");
    glUniformMatrix4fv(modelLocation, 1, GL_FALSE, glm::value_ptr(model));*/

    _buffer->Bind(false);
    _buffer->Draw();
}
