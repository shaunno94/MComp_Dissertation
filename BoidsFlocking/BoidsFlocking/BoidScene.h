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

	//k value
	const float MAX_DISTANCE = 65.0f;
	std::vector<Boid*> boids;

	std::function<float()> rndX;
	std::function<float()> rndY;
	std::function<float()> rndZ;

	float count = 0.0f;
	glm::vec3 m_FlockHeading;
	unsigned int maxBoids;

#if CUDA
	BoidGPU* boidsGPU;
	const uint32_t THREADS_PER_BLOCK = 1024;
	uint32_t BLOCKS_PER_GRID;
#endif
#if THREADED
	std::vector<std::future<void>> futures;
	const unsigned int NUMBER_OF_THREADS = 8;
	void UpdatePartition(size_t begin, size_t end, float dt);
#endif
};	
#if CUDA
__global__ void compute_KNN(BoidGPU* boid, const uint32_t maxBoids, const float MAX_DISTANCE);
__device__ void CalcCohesion(BoidGPU& boid, glm::vec3& cohVec);
__device__ void CalcSeperation(BoidGPU& boid, glm::vec3& sepVec);
__device__ void CalcAlignment(BoidGPU& boid, glm::vec3& alignVec);
__global__ void updateBoids(BoidGPU* boid, float dt, const uint32_t maxBoids);
#endif