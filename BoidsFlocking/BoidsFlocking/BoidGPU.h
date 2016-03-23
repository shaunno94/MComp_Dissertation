#pragma once
#include "Common.h"
#if CUDA
struct BoidGPU
{	
	glm::vec3 m_Position;
	glm::vec3 m_OldPosition;
	glm::vec3 m_Velocity;
	glm::vec3 m_CohesiveVector;
	glm::vec3 m_SeperationVector;
	glm::vec3 m_AlignmentVector;
	//glm::mat4 m_WorldTransform;
	BoidGPU** neighbours;
	unsigned int lastIndex;

	BoidGPU(unsigned int maxNeighbours, glm::vec3 spawnPos, glm::vec3 initialVelocity)
	{
		m_CohesiveVector = glm::vec3(0, 0, 0);
		m_AlignmentVector = glm::vec3(0, 0, 0);
		m_SeperationVector = glm::vec3(0, 0, 0);
		m_Position = spawnPos;
		m_Velocity = initialVelocity;
		m_OldPosition = spawnPos;
		//m_WorldTransform = glm::mat4(1.0f);
		cudaMalloc((void**)&neighbours, sizeof(int64_t) * maxNeighbours);
		lastIndex = 0;
	}
};
#endif