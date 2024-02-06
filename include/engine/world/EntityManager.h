#pragma once
#include <engine/world/Types.h>
#include <queue>
#include <array> 

/// <summary>
/// Entity Manager handles distributing entity IDs and keeping record of which IDs are in use and which are not.
/// </summary>
class EntityManager
{
public:
	EntityManager();
	Entity CreateEntity();
	void DestroyEntity(Entity entity);
	void SetSignature(Entity entity, Signature signature);
	Signature GetSignature(Entity entity);

private:
	/// Queue of unused entity IDs.
	std::queue<Entity> _availableEntities{};

	/// Array of signatures where the index corresponds to the entity ID.
	std::array<Signature, MAX_ENTITIES> _signatures{};

	/// Total living entities - used to keep limits on how many exist.
	int _livingEntityCount{};
};