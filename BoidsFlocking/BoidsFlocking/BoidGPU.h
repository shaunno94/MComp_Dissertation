#pragma once
#include "Common.h"
#if CUDA
//Used to store Boid data 
struct BoidGPU
{
	glm::vec3 m_Position[1250048];
	glm::vec3 m_Velocity[1250048];
	glm::vec3 m_CohesiveVector[1250048];
	glm::vec3 m_SeperationVector[1250048];
	glm::vec3 m_AlignmentVector[1250048];
	int m_Key[1250048];
	unsigned int m_Val[1250048];
};
#endif