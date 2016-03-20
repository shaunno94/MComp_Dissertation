#pragma once
#include "Common.h"
#if CUDA
struct BoidGPU
{
public:
	BoidGPU()
	{
		m_Position = glm::vec3(0, 0, 0);
		m_Velocity = glm::vec3(0, 0, 0);
		neighbours = nullptr;
		lastIndex = -1;
	}
	BoidGPU(unsigned int maxNeighbours, glm::vec3 spawnPos, glm::vec3 initialVelocity)
	{
		m_Position = spawnPos;
		m_Velocity = initialVelocity;
		neighbours = (BoidGPU*)malloc(sizeof(BoidGPU) * maxNeighbours);
		memset(neighbours, 0, sizeof(BoidGPU) * maxNeighbours);
		lastIndex = -1;
	}
	BoidGPU* neighbours;
	glm::vec3 m_Position;
	glm::vec3 m_Velocity;
	int lastIndex;
};
#endif