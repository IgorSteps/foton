//#include <engine/world/Entity.h>
//
//Entity::Entity(int id, const std::string& name, std::shared_ptr<Scene> scene) : _id(id), Name(name), _scene(scene)
//{
//	_worldMatrix = glm::mat4(1.0f);
//	_localMatrix = glm::mat4(1.0f);
//	Position = glm::vec3(1.0f, 0.0f, 1.0f);
//}
//
//int Entity::GetID() const
//{
//	return _id;
//}
//
//std::weak_ptr<Entity> Entity::GetParent() const 
//{
//	return _parent;
//}
//
//glm::mat4 Entity::GetWorldMatrix() const
//{
//	return _worldMatrix;
//}
//
//void Entity::AddChild(std::shared_ptr<Entity> child)
//{
//	child->OnAdded(_scene.lock()); // Lock to get a shared_ptr from weak_ptr
//	child->_parent = shared_from_this(); // Set parent to this entity
//	_children.push_back(std::move(child)); // Use move to transfer ownership
//}
//
//void Entity::OnAdded(std::shared_ptr<Scene> scene)
//{
//	_scene = scene;
//}
//
//void Entity::init()
//{
//	for (auto& c : _children)
//	{
//		c->init();
//	}
//}
//
//void Entity::update(float dt)
//{
//	for (auto& c : _children)
//	{
//		c->update(dt);
//	}
//}
//
//void Entity::draw()
//{
//	for (auto& c : _children)
//	{
//		c->draw();
//	}
//}
//
