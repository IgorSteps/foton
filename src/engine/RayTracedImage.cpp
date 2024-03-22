#include "engine/RayTracedImage.h"

RayTracedImage::RayTracedImage(PBO* pbo, float width, float height) :
	_width(width), _height(height), _pbo(pbo) {}

RayTracedImage::~RayTracedImage()
{
}

void RayTracedImage::Init()
{
	_sprite = std::make_unique<Sprite>();
	_texture = std::make_unique<Texture>(_width, _height);

	_sprite->Init();
	_texture->Init();
}

void RayTracedImage::Update(float width, float height)
{
	_pbo->Bind();
	_texture->Update(width, height);
	_pbo->Unbind();
}

void RayTracedImage::Draw(std::unique_ptr<Shader>& shader)
{
	_pbo->Bind();
	_texture->Draw(shader);
	_sprite->Draw(shader);
	_pbo->Unbind();
}
