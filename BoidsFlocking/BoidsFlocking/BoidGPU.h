#pragma once
#include "Common.h"
#if CUDA
struct BoidGPU;
struct Neighbour
{
	unsigned int n;
	float distance = 0.0f;
};

struct BoidGPU
{	
	glm::vec3 m_Position;
	glm::vec3 m_OldPosition;
	glm::vec3 m_Velocity;
	glm::vec3 m_CohesiveVector;
	glm::vec3 m_SeperationVector;
	glm::vec3 m_AlignmentVector;
	glm::quat m_Rotation;
	Neighbour* neighbours;
	unsigned int lastIndex;

	BoidGPU(unsigned int maxNeighbours, glm::vec3 spawnPos, glm::vec3 initialVelocity)
	{
		m_CohesiveVector = glm::vec3(0, 0, 0);
		m_AlignmentVector = glm::vec3(0, 0, 0);
		m_SeperationVector = glm::vec3(0, 0, 0);
		m_Rotation = glm::quat(0, 0, 0, 0);
		m_Position = spawnPos;
		m_Velocity = initialVelocity;
		m_OldPosition = spawnPos;
		cudaMalloc((void**)&neighbours, sizeof(Neighbour) * maxNeighbours);
		lastIndex = 0;
	}
};
#endif