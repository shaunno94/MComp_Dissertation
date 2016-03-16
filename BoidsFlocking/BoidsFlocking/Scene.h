#pragma once
#include <vector>
#include "Entity.h"
#include "Camera.h"

class Scene
{
public:
	Scene();
	virtual ~Scene();

	virtual void RenderScene();
	virtual void UpdateScene(float dt);

	Entity* GetOpaqueObject(unsigned int i);
	Entity* GetTransparentObject(unsigned int i);
	Camera* GetCamera() { return cam; }
	void AddEntity(Entity* e);

protected:
	std::vector<Entity*> transparentObjects;
	std::vector<Entity*> opaqueObjects;

	Camera* cam;
};