#include "engine/graphics/Sprite.h"

Sprite::Sprite(const std::string& name, float width, float height)
{
	_name = name;
	_width = width;
	_height = height;
}

Sprite::~Sprite()
{
    _buffer->~GLBuffer();
}

void Sprite::Init()
{

    // top left corner at 0,0
    std::vector<float>  vertices = {
      
            // positions        
             0.5f,  0.5f, 0.0f,  
             0.5f, -0.5f, 0.0f,  
            -0.5f, -0.5f, 0.0f,  
            -0.5f,  0.5f, 0.0f,   

    };

    std::vector<unsigned int> indices = {
        0, 1, 3,
        1, 2, 3
    };

    // Create a GLBuffer object
    _buffer = std::make_unique<GLBuffer>(GL_FLOAT, GL_TRIANGLES);

    // Define attribute information
    AttributeInfo positionAttrib(0, 3, 0);
    _buffer->AddAttributeLocation(positionAttrib);

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

void Sprite::Draw()
{
    _buffer->Bind(false);
    _buffer->Draw();
}
