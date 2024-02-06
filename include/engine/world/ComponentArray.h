#pragma once
#include <engine/world/Types.h>

class IComponentArray
{
public:
	virtual ~IComponentArray() = default;
	virtual void EntityDestroyed(Entity entity) = 0;
};

template<typename T>
class ComponentArray : public IComponentArray
{
public:
	void InsertData(Entity entity, T component)
	{
		assert(_entityToIndexMap.find(entity) == _entityToIndexMap.end() && "Component added to same entity more than once.");

		// Put new entry at end and update the maps
		size_t newIndex = mSize;
		_entityToIndexMap[entity] = newIndex;
		_indexToEntityMap[newIndex] = entity;
		_componentArray[newIndex] = component;
		++mSize;
	}

	void RemoveData(Entity entity)
	{
		assert(_entityToIndexMap.find(entity) != _entityToIndexMap.end() && "Removing non-existent component.");

		// Copy element at end into deleted element's place to maintain density
		size_t indexOfRemovedEntity = _entityToIndexMap[entity];
		size_t indexOfLastElement = mSize - 1;
		_componentArray[indexOfRemovedEntity] = _componentArray[indexOfLastElement];

		// Update map to point to moved spot
		Entity entityOfLastElement = _indexToEntityMap[indexOfLastElement];
		_entityToIndexMap[entityOfLastElement] = indexOfRemovedEntity;
		_indexToEntityMap[indexOfRemovedEntity] = entityOfLastElement;

		_entityToIndexMap.erase(entity);
		_indexToEntityMap.erase(indexOfLastElement);

		--mSize;
	}

	T& GetData(Entity entity)
	{
		assert(_entityToIndexMap.find(entity) != _entityToIndexMap.end() && "Retrieving non-existent component.");

		// Return a reference to the entity's component
		return _componentArray[_entityToIndexMap[entity]];
	}

	void EntityDestroyed(Entity entity) override
	{
		if (_entityToIndexMap.find(entity) != _entityToIndexMap.end())
		{
			// Remove the entity's component if it existed
			RemoveData(entity);
		}
	}

private:
	/// The packed array of components (of generic type T),
	/// set to a specified maximum amount, matching the maximum number
	/// of entities allowed to exist simultaneously, so that each entity
	/// has a unique spot.
	std::array<T, MAX_ENTITIES> _componentArray;

	/// Map from an entity ID to an array index.
	std::unordered_map<Entity, size_t> _entityToIndexMap;

	/// Map from an array index to an entity ID.
	std::unordered_map<size_t, Entity> _indexToEntityMap;

	/// Total size of valid entries in the array.
	size_t _size;
};