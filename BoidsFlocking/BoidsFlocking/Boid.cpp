#include "Boid.h"
#include <algorithm> 

const float Boid::MIN_DIST = 6.0f;
const float Boid::MAX_SPEED = 0.25f;
const unsigned int Boid::K = 65;

Boid::Boid(unsigned int maxBoids, glm::vec3 spawnPosition, glm::vec3 initialVelocity, const std::string& name) : Entity(name)
{
	m_Position = spawnPosition;
	m_Velocity = initialVelocity;
	m_Heading = glm::vec3(0, 0, 0);
	m_Destination = glm::vec3(0, 0, 0);
	neighbours.clear();

	//neighbours.resize(maxBoids - 1);
	maxSize = maxBoids - 1;
}

Boid::~Boid()
{

}

void Boid::OnUpdateObject(float dt)
{
	std::sort(neighbours.begin(), neighbours.begin() + lastPosition, comp);
	CalculateForce();

	//m_Velocity = m_Velocity + (accel * dt);
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
	m_Velocity = m_SeperationVector + m_CohesiveVector + m_AlignmentVector + m_Heading;
}

void Boid::CalcCohesion()
{
	unsigned int counter = 0;
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		//if (neighbours[i].dist < MIN_DIST)
		{
			avgPos += neighbours[i].n->m_Position;
			counter++;
		}
	}
	//if (counter > 0)
	{
		//avgPos /= float(counter);
		avgPos /= float(K);
		m_CohesiveVector = (avgPos - m_Position) /* 0.001f*/;

		float mag = glm::length(m_CohesiveVector);
		glm::normalize(m_CohesiveVector);

		//if (mag < 10.0f)
		{
			m_CohesiveVector *= (MAX_SPEED * (mag * 0.001f));
		}
		/*else
		{
			m_CohesiveVector *= MAX_SPEED;
		}*/

		m_CohesiveVector -= m_Velocity;
	}	
	/*else
	{
		m_CohesiveVector = glm::vec3(0, 0, 0);
	}*/
}

void Boid::CalcSeperation()
{
	unsigned int counter = 0;
	glm::vec3 seperation = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		//if (neighbours[i].dist < MIN_DIST)
		{
			//seperation -= (neighbours[i].n->m_Position - m_Position);
			seperation -= (neighbours[i].n->m_Position - m_Position) / neighbours[i].dist;
			counter++;
		}
	}
	//if (counter > 0)
		seperation /= float(K);

	m_SeperationVector = seperation * 0.3f;
}

void Boid::CalcAlignment()
{
	unsigned int counter = 0;
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < K; ++i)
	{
		//if (neighbours[i].dist < 60.0f)
		{
			avgVel += neighbours[i].n->m_Velocity;
			counter++;
		}
	}
	//if (counter > 0)
		//avgVel /= float(counter);
	avgVel /= K;
	m_AlignmentVector = (avgVel - m_Velocity) * 0.8f;
}