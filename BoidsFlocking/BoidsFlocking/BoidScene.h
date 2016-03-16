#pragma once
#include "Scene.h"
#include "Boid.h"

class BoidScene : public Scene
{
public: 
	BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh);
	virtual ~BoidScene();
	virtual void UpdateScene(float dt) override;

protected:

private:
	BoidScene() {}
};