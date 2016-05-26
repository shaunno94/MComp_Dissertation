#pragma once
#include "Common.h"
#if CUDA
//Used to store Boid data 
struct BoidGPU
{
	glm::vec3* m_Position;
	glm::vec3* m_Velocity;
	glm::vec3* m_CohesiveVector;
	glm::vec3* m_SeperationVector;
	glm::vec3* m_AlignmentVector;
	int* m_Key;
	unsigned int* m_Val;
};
#endif