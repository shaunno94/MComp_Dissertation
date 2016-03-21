#include "Boid.h"
#include <algorithm> 

const float Boid::MIN_DIST = 6.0f;
const float Boid::MAX_SPEED = 0.3f;
const unsigned int Boid::K = 65;

Boid::Boid(unsigned int maxBoids, glm::vec3 spawnPosition, glm::vec3 initialVelocity, const std::string& name) : Entity(name)
{
	m_Position = spawnPosition;
	m_Velocity = initialVelocity;
	m_Heading = glm::vec3(0, 0, 0);
	m_Destination = glm::vec3(0, 0, 0);
	neighbours.clear();
}

Boid::~Boid()
{

}

void Boid::OnUpdateObject(float dt)
{
	CalculateForce();

	m_OldPosition = m_Position;
	m_Position += (m_Velocity * dt);
	
	m_WorldTransform = glm::translate(glm::mat4(1.0f), m_OldPosition + m_Position);
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

void Boid::TendToPlace()
{
	m_Heading = (m_Destination - m_Position) * 0.001f;
}

void Boid::CalculateForce()
{
	CalcCohesion();
	CalcSeperation();
	CalcAlignment();
	TendToPlace();
	m_Velocity = (m_SeperationVector + m_CohesiveVector + m_AlignmentVector + m_Heading);
	LimitVelocity();
}

void Boid::CalcCohesion()
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < neighbours.size(); ++i)
	{
		avgPos += neighbours[i].n->m_Position;
	}
	avgPos /= float(neighbours.size());
	m_CohesiveVector = (avgPos - m_Position);

	float mag = glm::length(m_CohesiveVector);
	glm::normalize(m_CohesiveVector);
	m_CohesiveVector *= (MAX_SPEED * (mag * 0.001f));
	m_CohesiveVector -= m_Velocity;
}

void Boid::CalcSeperation()
{
	glm::vec3 seperation = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < neighbours.size(); ++i)
	{
		seperation -= (neighbours[i].n->m_Position - m_Position) / neighbours[i].dist;
	}
	seperation /= float(neighbours.size());
	m_SeperationVector = seperation * 0.25f;
}

void Boid::CalcAlignment()
{
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < neighbours.size(); ++i)
	{
		avgVel += neighbours[i].n->m_Velocity;
	}
	avgVel /= float(neighbours.size());
	m_AlignmentVector = (avgVel - m_Velocity);
}