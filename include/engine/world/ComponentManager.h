#pragma once
#include <unordered_map>
#include <memory>
#include <engine/world/ComponentArray.h>

class ComponentManager
{
public:
	template<typename T>
	void RegisterComponent()
	{
		const char* typeName = typeid(T).name();

		assert(_componentTypes.find(typeName) == _componentTypes.end() && "Registering component type more than once.");

		// Add this component type to the component type map
		_componentTypes.insert({ typeName, _nextComponentType });

		// Create a ComponentArray pointer and add it to the component arrays map
		_componentArrays.insert({ typeName, std::make_shared<ComponentArray<T>>() });

		// Increment the value so that the next component registered will be different
		++_nextComponentType;
	}

	template<typename T>
	ComponentType GetComponentType()
	{
		const char* typeName = typeid(T).name();

		assert(_componentTypes.find(typeName) != _componentTypes.end() && "Component not registered before use.");

		// Return this component's type - used for creating signatures
		return _componentTypes[typeName];
	}

	template<typename T>
	void AddComponent(Entity entity, T component)
	{
		// Add a component to the array for an entity
		GetComponentArray<T>()->InsertData(entity, component);
	}

	template<typename T>
	void RemoveComponent(Entity entity)
	{
		// Remove a component from the array for an entity
		GetComponentArray<T>()->RemoveData(entity);
	}

	template<typename T>
	T& GetComponent(Entity entity)
	{
		// Get a reference to a component from the array for an entity
		return GetComponentArray<T>()->GetData(entity);
	}

	void EntityDestroyed(Entity entity)
	{
		// Notify each component array that an entity has been destroyed
		// If it has a component for that entity, it will remove it
		for (auto const& pair : _componentArrays)
		{
			auto const& component = pair.second;

			component->EntityDestroyed(entity);
		}
	}

private:
	/// Map from type string pointer to a component type
	std::unordered_map<const char*, ComponentType> _componentTypes{};

	/// Map from type string pointer to a component array
	std::unordered_map<const char*, std::shared_ptr<IComponentArray>> _componentArrays{};

	/// The component type to be assigned to the next registered component - starting at 0
	ComponentType _nextComponentType{};

	/// Convenience function to get the statically casted pointer to the ComponentArray of type T.
	template<typename T>
	std::shared_ptr<ComponentArray<T>> getComponentArray()
	{
		const char* typeName = typeid(T).name();

		assert(_componentTypes.find(typeName) != _componentTypes.end() && "Component not registered before use.");

		return std::static_pointer_cast<ComponentArray<T>>(_componentArrays[typeName]);
	}
};