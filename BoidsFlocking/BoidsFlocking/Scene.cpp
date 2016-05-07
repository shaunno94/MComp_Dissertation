#include "Scene.h"
#include "OGLRenderer.h"

Scene::Scene()
{
	cam = new Camera(OGLRenderer::Instance()->GetWindow(), OGLRenderer::Instance()->GetWindowHeight(), OGLRenderer::Instance()->GetWindowWidth(), 45.0f, 0.1f, 10000.0f);
}

Scene::~Scene()
{
	for (Entity* e : opaqueObjects)
	{
		delete e;
		e = nullptr;
	}
	for (Entity* e : transparentObjects)
	{
		delete e;
		e = nullptr;
	}
	opaqueObjects.clear();
	transparentObjects.clear();
}

Entity* Scene::GetOpaqueObject(unsigned int i)
{
	if (i < opaqueObjects.size())
		return opaqueObjects[i];
	else
		return nullptr;
}

Entity* Scene::GetTransparentObject(unsigned int i)
{
	if (i < transparentObjects.size())
		return transparentObjects[i];
	else
		return nullptr;
}

void Scene::AddEntity(Entity* e)
{
	opaqueObjects.push_back(e);
	for (auto child : e->GetChildren())
	{
		AddEntity(child);
	}
}

void Scene::UpdateScene(float dt)
{
	cam->UpdateCamera(dt);

	for (unsigned int i = 0; i < opaqueObjects.size(); ++i)
		opaqueObjects[i]->OnUpdateObject(dt);

	for (unsigned int i = 0; i < transparentObjects.size(); ++i)
		transparentObjects[i]->OnUpdateObject(dt);
}

void Scene::RenderScene()
{
	for (unsigned int i = 0; i < opaqueObjects.size(); ++i)
		opaqueObjects[i]->OnRenderObject(i);

	for (unsigned int i = 0; i < transparentObjects.size(); ++i)
		transparentObjects[i]->OnRenderObject(i);
}