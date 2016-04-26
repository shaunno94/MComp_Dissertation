#pragma once
#include "Common.h"
#if CUDA

/*struct BoidGPU
{	
	glm::vec3 m_Position;
	glm::vec3 m_Velocity;
	glm::vec3 m_CohesiveVector;
	glm::vec3 m_SeperationVector;
	glm::vec3 m_AlignmentVector;
	glm::quat m_Rotation;

	BoidGPU(glm::vec3 spawnPos, glm::vec3 initialVelocity)
	{
		m_CohesiveVector = glm::vec3(0, 0, 0);
		m_AlignmentVector = glm::vec3(0, 0, 0);
		m_SeperationVector = glm::vec3(0, 0, 0);
		m_Rotation = glm::quat(0, 0, 0, 0);
		m_Position = spawnPos;
		m_Velocity = initialVelocity;
	}
};*/

struct BoidGPU
{
	glm::vec3 m_Position[NUM_BOIDS];
	glm::vec3 m_Velocity[NUM_BOIDS];
	glm::vec3 m_CohesiveVector[NUM_BOIDS];
	glm::vec3 m_SeperationVector[NUM_BOIDS];
	glm::vec3 m_AlignmentVector[NUM_BOIDS];
	glm::quat m_Rotation[NUM_BOIDS];
};
#endif