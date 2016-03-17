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
	Boid(unsigned int maxBoids, glm::vec3 spawnPosition, glm::vec3 initialVelocity, const std::string& name = std::to_string(id));
	virtual ~Boid();

	inline const glm::vec3& GetPosition() const { return m_Position; }
	inline void UpdateFlockHeading(glm::vec3& heading) { m_Destination = heading; }
	inline const glm::vec3& GetVelocity() const { return m_Velocity; }
	inline void AddNeighbour(BoidNeighbour bN) { neighbours[lastPosition++] = bN; }

protected:
	virtual void OnUpdateObject(float dt) override;

private:
	void CalculateForce();
	void CalcCohesion();
	void CalcSeperation();
	void CalcAlignment();
	void LimitVelocity();
	void TendToPlace();

	glm::vec3 m_Destination;
	glm::vec3 m_Heading;
	glm::vec3 m_Position;
	glm::vec3 m_Velocity;
	glm::vec3 m_CohesiveVector;
	glm::vec3 m_SeperationVector;
	glm::vec3 m_AlignmentVector;
	std::vector<BoidNeighbour> neighbours;

	static const float MIN_DIST;
	static const float MAX_SPEED;
	static const unsigned int K;

	unsigned int lastPosition = 0;
	unsigned int maxSize;
	
	struct compare
	{
		bool operator()(const BoidNeighbour& a, const BoidNeighbour& b) { return a.dist < b.dist; }
	} comp;
};