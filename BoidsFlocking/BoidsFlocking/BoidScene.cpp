#include "BoidScene.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

BoidScene::BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh)
{
	for (unsigned int i = 0; i < numberOfBoids; ++i)
	{
		Boid* b = new Boid();
		RenderComponent* rc = new RenderComponent(mesh, shader);
		b->SetRenderComponent(rc);
		AddEntity(b);
	}
}

BoidScene::~BoidScene()
{

}

void BoidScene::UpdateScene(float dt)
{
	//glm::length(glm::vec3());
	Scene::UpdateScene(dt);
}