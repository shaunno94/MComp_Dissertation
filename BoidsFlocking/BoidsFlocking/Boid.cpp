#include "Boid.h"
#include <algorithm> 

const float Boid::MIN_DIST = 5.0f;
const float Boid::MAX_SPEED = 0.01f;
const unsigned int Boid::K = 12;

Boid::Boid(unsigned int maxBoids, glm::vec3 spawnPosition, const std::string& name) : Entity(name)
{
	m_Position = spawnPosition;
	m_Velocity = glm::vec3(0, 0, 0);
	m_Force = glm::vec3(0, 0, 0);
	m_InvMass = 1.0f;
	neighbours.clear();

	neighbours.resize(maxBoids - 1);
	maxSize = maxBoids - 1;
}

Boid::~Boid()
{

}

void Boid::OnUpdateObject(float dt)
{
	std::sort(neighbours.begin(), neighbours.begin() + lastPosition, comp);
	CalculateForce();

	glm::vec3 accel = m_Force * m_InvMass;
	m_Velocity = m_Velocity + (accel * dt);
	LimitVelocity();
	m_Position = m_Position + (m_Velocity * dt);

	m_WorldTransform = glm::translate(glm::mat4(1.0f), m_Position);
	m_WorldTransform = m_WorldTransform * m_LocalTransform;

	lastPosition = 0;
}

void Boid::LimitVelocity()
{
	float speed = glm::length(m_Velocity);
	if (speed > MAX_SPEED)
	{
		m_Velocity = (m_Velocity / speed) * MAX_SPEED;
	}
}

void Boid::CalculateForce()
{
	CalcCohesion();
	CalcSeperation();
	CalcAlignment();
	m_Force = m_CohesiveForce + m_SeperationForce + m_AlignmentForce;
}

void Boid::CalcCohesion()
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		avgPos += neighbours[i].n->m_Position;
	}
	avgPos /= K;
	m_CohesiveForce = (avgPos - m_Position) / 100.0f;
}

void Boid::CalcSeperation()
{
	glm::vec3 seperation = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		if (neighbours[i].dist < MIN_DIST)
		{
			seperation -= (neighbours[i].n->m_Position - m_Position);
		}
	}
	m_SeperationForce = seperation;
}

void Boid::CalcAlignment()
{
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		avgVel += neighbours[i].n->m_Velocity;
	}
	avgVel /= K;
	m_AlignmentForce = (avgVel - m_Velocity) / 8.0f;
}