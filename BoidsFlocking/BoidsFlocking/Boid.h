#pragma once
#include "Entity.h"

class Boid : public Entity
{
	friend class Scene;
public:
	Boid(const std::string& name = std::to_string(id));
	virtual ~Boid();

protected:		
	virtual void OnUpdateObject(float dt) override;

private:
	glm::vec3 m_Position;
	glm::vec3 m_OldPosition;
	glm::vec3 m_Velocity;
	glm::vec3 m_Force;
	float m_InvMass;
};