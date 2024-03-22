#pragma once
#include <memory>
#include <engine/graphics/Sprite.h>
#include <engine/graphics/Texture.h>
#include <engine/gl/PBOBuffer.h>

class RayTracedImage
{
public:
	RayTracedImage(PBO* pbo, float width, float height);
	~RayTracedImage();

	void Init();
	void Update(float width, float height);
	void Draw(std::unique_ptr<Shader>& shader);

private:
	float _width, _height;
	PBO* _pbo;

	std::unique_ptr<Sprite> _sprite;
	std::unique_ptr<Texture> _texture;
};