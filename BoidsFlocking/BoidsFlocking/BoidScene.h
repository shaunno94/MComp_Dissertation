#pragma once
#include "Scene.h"
#include "Boid.h"
#include <future>
#include <random>
#include <functional>

class BoidScene : public Scene
{
public: 
	BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh);
	virtual ~BoidScene();
	virtual void UpdateScene(float dt) override;
	Boid* GetBoid(unsigned int i) { return i < boids.size() ? boids[i] : nullptr; }
	std::vector<Boid*>& GetBoidData() { return boids; }

protected:

private:
	BoidScene() {}
	void UpdatePartition(unsigned int begin, unsigned int end, float dt);
	void InitGenerator(int spread);

	//k value
	const float MAX_DISTANCE = 90.0f;
	std::vector<Boid*> boids;

	std::function<float()> rndX;
	std::function<float()> rndY;
	std::function<float()> rndZ;

	float count = 0.0f;
	glm::vec3 m_FlockHeading;

#if THREADED
	std::vector<std::future<void>> futures;
	const unsigned int NUMBER_OF_THREADS = 8;
#endif
};