#pragma once
#include "Scene.h"
#include "Boid.h"
#include <future>
#include <random>
#include <functional>
#include "Common.h"
#if CUDA
#include "BoidGPU.h"
#endif

class BoidScene : public Scene
{
public: 
	BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh);
	virtual ~BoidScene();
	virtual void UpdateScene(float dt) override;
	virtual void RenderScene() override;
	Boid* GetBoid(unsigned int i) { return i < boids.size() ? boids[i] : nullptr; }
	std::vector<Boid*>& GetBoidData() { return boids; }

protected:

private:
	BoidScene() {}
	void InitGenerator(int spread);

	std::vector<Boid*> boids;

	std::function<float()> rndX;
	std::function<float()> rndY;
	std::function<float()> rndZ;

	float count = 0.0f;
	glm::vec3 m_FlockHeading;

#if CUDA
	BoidGPU* boidsHost;
	BoidGPU* boidsDevice;
	glm::mat4* modelMatricesHostPinned;
	glm::mat4* modelMatricesDevice;
	uint32_t BLOCKS_PER_GRID;
#endif
#if THREADED
	std::vector<std::future<void>> futures;
	const unsigned int NUMBER_OF_THREADS = 8;
	void UpdatePartition(size_t begin, size_t end, float dt);
#endif
};	
#if CUDA
__global__ void ComputeKNN(BoidGPU* boid);
__global__ void CalcVelocity(BoidGPU* boid, const glm::vec3 heading);
__global__ void UpdateBoid(BoidGPU* boid, glm::mat4* boidMat, const float dt);
#endif