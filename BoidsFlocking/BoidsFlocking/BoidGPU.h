#pragma once
#include "Common.h"
#if CUDA

struct BoidGPU
{
	glm::vec3 m_Position[NUM_BOIDS];
	glm::vec3 m_Velocity[NUM_BOIDS];
	glm::vec3 m_CohesiveVector[NUM_BOIDS];
	glm::vec3 m_SeperationVector[NUM_BOIDS];
	glm::vec3 m_AlignmentVector[NUM_BOIDS];
	unsigned int m_Key[NUM_BOIDS];
	unsigned int m_Val[NUM_BOIDS];
};
#endif