#include "Boid.h"
#include <algorithm> 
#define K 450

const float Boid::MAX_SPEED = 12.0f;
const float Boid::MAX_SPEED_SQR = MAX_SPEED * MAX_SPEED;
glm::vec3 Boid::m_Heading = glm::vec3(0, 0, 0);

Boid::Boid(glm::vec3 spawnPosition, glm::vec3 initialVelocity, const std::string& name) : Entity(name)
{
	m_Position = spawnPosition;
	m_Velocity = initialVelocity;
	m_WorldTransformPtr = &m_WorldTransform;
	neighbours.clear();
}

Boid::~Boid()
{

}

void Boid::OnUpdateObject(float dt)
{
	std::partial_sort(neighbours.begin(), neighbours.begin() + K, neighbours.end(), comp);
	
	CalculateVelocity(dt);

	m_Position += (m_Velocity * (1.0f / dt));
	
	m_WorldTransform = glm::mat4_cast(glm::quat(m_Velocity)) * glm::translate(m_Position);
	m_WorldTransform = m_WorldTransform * m_LocalTransform;
	neighbours.clear();
}

void Boid::CalculateVelocity(float dt)
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	glm::vec3 seperation = glm::vec3(0, 0, 0);
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	float avgDiv = 1.0f / float(K);

	//Loop through neighbours
	for (unsigned int i = 0; i < K; ++i)
	{
		avgPos += neighbours[i].n->m_Position;
		seperation -= (neighbours[i].n->m_Position - m_Position) * (1.0f / sqrtf(neighbours[i].dist));
		avgVel += neighbours[i].n->m_Velocity;
	}

	//Calculate Cohesion
	avgPos *= avgDiv;
	m_CohesiveVector = (avgPos - m_Position);

	float mag = glm::length(m_CohesiveVector);
	m_CohesiveVector = glm::normalize(m_CohesiveVector);
	m_CohesiveVector *= (MAX_SPEED * (mag * 0.001f));
	m_CohesiveVector -= m_Velocity;

	//Calculate Seperation
	seperation *= avgDiv;
	m_SeperationVector *= 0.25f;

	//Calculate Alignment
	avgVel *= avgDiv;
	m_AlignmentVector = (avgVel - m_Velocity);

	//Calculate final velocity
	m_Velocity += (m_SeperationVector + m_CohesiveVector + m_AlignmentVector + 
		(glm::cross(m_Heading, m_Position) * 0.05f)) * (1.0f / dt);

	float speed = glm::dot(m_Velocity, m_Velocity);
	if (speed > MAX_SPEED_SQR)
	{
		m_Velocity = (m_Velocity * (1.0f / sqrtf(speed))) * MAX_SPEED;
	}
	m_Velocity *= m_DampingFactor;
}