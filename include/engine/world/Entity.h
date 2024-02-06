//#pragma once
//
//#include <vector>
//#include <string>
//#include <glm/glm.hpp>
//#include <memory>
//class Scene;
//
//class Entity : std::enable_shared_from_this<Entity>
//{
//public:
//	Entity(int id, const std::string& name, std::shared_ptr<Scene> scene);
//	//~Entity();
//
//	int GetID() const;
//	std::weak_ptr<Entity> GetParent() const;
//	glm::mat4 GetWorldMatrix() const; 
//
//	void AddChild(std::shared_ptr<Entity> child);
//	void OnAdded(std::shared_ptr<Scene> scene);
//
//	void init();
//	void update(float dt);
//	void draw();
//
//	std::string Name;
//	glm::vec3 Position;
//
//private:
//	int _id;
//	std::vector<std::shared_ptr<Entity>> _children;
//	glm::mat4 _localMatrix;
//	glm::mat4 _worldMatrix;
//
//	// Weak pointers to avoid cyclic references, ensuring that parent entities
//	// do not prevent their children from being freed and vice versa.
//	std::weak_ptr<Entity> _parent; 
//	std::weak_ptr<Scene> _scene; // Scene reference should also be weak to avoid ownership cycles
//};