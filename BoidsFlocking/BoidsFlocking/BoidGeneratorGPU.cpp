#include "BoidGeneratorGPU.h"
#if CUDA
#include <iostream>
BoidGeneratorGPU::BoidGeneratorGPU(unsigned int numberOfBoids)
{
	InitGenerator(numberOfBoids);

	cudaMallocManaged((void**)&boids, numberOfBoids * sizeof(BoidGPU));

	for (unsigned int i = 0; i < numberOfBoids; ++i)
	{
		boids[i] = BoidGPU(50, glm::vec3(rndX(), rndY(), rndZ()), glm::vec3(rndX(), rndY(), rndZ()));
	}
	m_FlockHeading = glm::vec3(0, 0, 0);
}

BoidGeneratorGPU::~BoidGeneratorGPU()
{
	cudaFree(boids);
}

void BoidGeneratorGPU::InitGenerator(int spread)
{
	std::random_device rD0;
	std::random_device rD1;
	std::random_device rD2;
	std::default_random_engine engine0(rD0());
	std::default_random_engine engine1(rD1());
	std::default_random_engine engine2(rD2());

	std::uniform_real_distribution<float> x(-100.0f, 100.0f);
	std::uniform_real_distribution<float> y(-100.0f, 100.0f);
	std::uniform_real_distribution<float> z(-100.0f, 100.0f);

	rndX = std::bind(x, engine0);
	rndY = std::bind(y, engine1);
	rndZ = std::bind(z, engine2);
}
#endif