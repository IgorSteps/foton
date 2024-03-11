#pragma once
#include <memory>
#include <engine/graphics/Sprite.h>
#include <engine/graphics/Texture.h>
#include <engine/gl/PBOBuffer.h>

class RayTracedImage
{
public:
	RayTracedImage(float width, float height);
	~RayTracedImage();

	void Init();
	void Update();
	void Draw(std::unique_ptr<Shader>& shader);

	int GetPBOID() const {
		return _pbo->GetID();
	}

private:
	float _width, _height;

	std::unique_ptr<Sprite> _sprite;
	std::unique_ptr<Texture> _texture;
	std::unique_ptr<PBO> _pbo;
};