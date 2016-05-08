#pragma once
#include "Entity.h"
#include "Common.h"

class Boid;
struct BoidNeighbour
{
	Boid* n = nullptr;
	float dist = 0.0f;
};

class Boid : public Entity
{
	friend class BoidScene;
public:
	Boid(glm::vec3 spawnPosition, glm::vec3 initialVelocity, const std::string& name = std::to_string(id));
	virtual ~Boid();

	inline const glm::vec3& GetPosition() const { return m_Position; }
	static void UpdateFlockHeading(glm::vec3& heading) { m_Heading = heading; }
	inline const glm::vec3& GetVelocity() const { return m_Velocity; }
	inline void AddNeighbour(BoidNeighbour bN) { neighbours.push_back(bN); }

protected:
	virtual void OnUpdateObject(float dt) override;

private:
	void CalculateVelocity(float dt);
	void LimitVelocity();

	glm::vec3 m_Position;
	glm::vec3 m_Velocity;
	glm::vec3 m_CohesiveVector;
	glm::vec3 m_SeperationVector;
	glm::vec3 m_AlignmentVector;
	const float m_DampingFactor = 0.999f;
	std::vector<BoidNeighbour> neighbours;

	static const float MAX_SPEED;
	static glm::vec3 m_Heading;
	
	struct compare
	{
		bool operator()(const BoidNeighbour& a, const BoidNeighbour& b) { return a.dist < b.dist; }
	} comp;
};