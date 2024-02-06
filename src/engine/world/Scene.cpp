//#include <engine/world/Scene.h>
//
//Scene::Scene()
//{
//	_root = std::make_shared<Entity>(0, "Root", shared_from_this());
//}
//
//std::shared_ptr<Entity> Scene::GetRoot() const
//{
//	return _root;
//}
//
//void Scene::AddEntity(const std::shared_ptr<Entity>& entity)
//{
//	_root->AddChild(entity);
//}
//
//void Scene::init()
//{
//	_root->init();
//}
//
//void Scene::update(float dt)
//{
//	_root->update(dt);
//}
//
//void Scene::draw()
//{
//	_root->draw();
//}
