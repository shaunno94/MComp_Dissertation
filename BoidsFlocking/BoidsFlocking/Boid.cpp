#include "Boid.h"
#include <algorithm> 

const float Boid::MAX_SPEED = 0.3f;
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
	CalculateVelocity();

	m_OldPosition = m_Position;
	m_Position += (m_Velocity * dt);
	
	m_WorldTransform = glm::mat4_cast(glm::quat(glm::vec3(m_Velocity.x, m_Velocity.y, m_Velocity.z))) * glm::translate(glm::mat4(1.0f), m_OldPosition + m_Position);
	m_WorldTransform = m_WorldTransform * m_LocalTransform;
	neighbours.clear();
}

void Boid::LimitVelocity()
{
	float speed = glm::length(m_Velocity);
	if (speed > MAX_SPEED)
	{
		m_Velocity = (m_Velocity / speed) * MAX_SPEED;
	}
	m_Velocity *= m_DampingFactor;
}

void Boid::CalculateVelocity()
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	glm::vec3 seperation = glm::vec3(0, 0, 0);
	glm::vec3 avgVel = glm::vec3(0, 0, 0);

	//Loop through neighbours
	for (unsigned int i = 0; i < neighbours.size(); ++i)
	{
		avgPos += neighbours[i].n->m_Position;
		seperation -= (neighbours[i].n->m_Position - m_Position) / neighbours[i].dist;
		avgVel += neighbours[i].n->m_Velocity;
	}

	//Calculate Cohesion
	avgPos /= float(neighbours.size());
	m_CohesiveVector = (avgPos - m_Position);

	float mag = glm::length(m_CohesiveVector);
	m_CohesiveVector = glm::normalize(m_CohesiveVector);
	m_CohesiveVector *= (MAX_SPEED * (mag * 0.001f));
	m_CohesiveVector -= m_Velocity;

	//Calculate Seperation
	seperation /= float(neighbours.size());
	m_SeperationVector *= 0.25f;

	//Calculate Alignment
	avgVel /= float(neighbours.size());
	m_AlignmentVector = (avgVel - m_Velocity);

	//Calculate final velocity
	m_Velocity = m_SeperationVector + m_CohesiveVector + m_AlignmentVector + 
		((m_Heading - m_Position) * 0.001f);

	LimitVelocity();
}