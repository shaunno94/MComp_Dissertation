#include "Boid.h"

Boid::Boid(const std::string& name) : Entity(name)
{
	m_Position = glm::vec3(0, 0, 0);
	m_OldPosition = glm::vec3(0, 0, 0);
	m_Velocity = glm::vec3(0, 0, 0);
	m_Force = glm::vec3(0, 0, 0);
	m_InvMass = 1.0f;
}

Boid::~Boid()
{

}

void Boid::OnUpdateObject(float dt)
{
	glm::vec3 accel = m_Force * m_InvMass;
	m_Velocity = m_Velocity + (accel * dt);
	m_OldPosition = m_Position;
	m_Position = m_Position + (m_Velocity * dt);

	m_WorldTransform = glm::translate(m_WorldTransform, (m_OldPosition - m_Position));
	m_WorldTransform = m_WorldTransform * m_LocalTransform;
}