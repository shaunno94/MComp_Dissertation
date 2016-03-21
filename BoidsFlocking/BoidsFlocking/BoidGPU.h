#pragma once
#include "Common.h"
#if CUDA
struct BoidGPU
{
	BoidGPU** neighbours;
	glm::vec3 m_Position;
	glm::vec3 m_Velocity;
	glm::mat4 m_WorldTransform = glm::mat4(1.0f);
	unsigned int lastIndex = 0;

	BoidGPU()
	{
		m_Position = glm::vec3(0, 0, 0);
		m_Velocity = glm::vec3(0, 0, 0);
		m_WorldTransform = glm::mat4(1.0f);
		neighbours = nullptr;
		lastIndex = 0;
	}
	BoidGPU(unsigned int maxNeighbours, glm::vec3 spawnPos, glm::vec3 initialVelocity)
	{
		m_Position = spawnPos;
		m_Velocity = initialVelocity;
		m_WorldTransform = glm::mat4(1.0f);
		cudaMalloc((void**)&neighbours, sizeof(BoidGPU) * maxNeighbours);
		lastIndex = 0;
	}
};
#endif