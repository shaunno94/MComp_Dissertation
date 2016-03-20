#pragma once
#include "Common.h"
#if CUDA
#include "BoidGPU.h"
#include <future>
#include <random>
#include <functional>

class BoidGeneratorGPU
{
public:
	BoidGeneratorGPU(unsigned int numberOfBoids);
	~BoidGeneratorGPU();
	BoidGPU* GetBoidData() { return boids; }

private:
	void InitGenerator(int spread);

	//k value
	const float MAX_DISTANCE = 90.0f;
	BoidGPU* boids;
	float count = 0.0f;
	glm::vec3 m_FlockHeading;

	std::function<float()> rndX;
	std::function<float()> rndY;
	std::function<float()> rndZ;
};
#endif