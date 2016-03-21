#include "BoidScene.h"
#include "Common.h"
#include <iostream>

BoidScene::BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh)
{
	InitGenerator(numberOfBoids);

#if CUDA
	cudaSetDevice(0);
	cudaMallocManaged((void**)&boidsGPU, numberOfBoids * sizeof(BoidGPU));
	BLOCKS_PER_GRID = (numberOfBoids + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
#define MAX_NEIGHBOURS 1024
#endif

	glm::vec3 pos, vel;
	for (unsigned int i = 0; i < numberOfBoids; ++i)
	{
		pos = glm::vec3(rndX(), rndY(), rndZ());
		vel = glm::vec3(rndX(), rndY(), rndZ());
		Boid* b = new Boid(pos, vel);
		b->SetRenderComponent(new RenderComponent(mesh, shader));
		boids.push_back(b);

#if CUDA
		boidsGPU[i] = BoidGPU(numberOfBoids, pos, vel);
//	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
	}
	m_FlockHeading = glm::vec3(0, 0, 0);
#if THREADED
	futures.clear();
#endif
	maxBoids = numberOfBoids;
}

BoidScene::~BoidScene()
{

}

void BoidScene::InitGenerator(int spread)
{
	std::random_device rD0;
	std::random_device rD1;
	std::random_device rD2;
	std::default_random_engine engine0(rD0());
	std::default_random_engine engine1(rD1());
	std::default_random_engine engine2(rD2());

	std::uniform_real_distribution<float> x(-100.0f, 100.0f);
	std::uniform_real_distribution<float> y(-100.0f, 100.0f);
	std::uniform_real_distribution<float> z(-100.0f, 100.0f);

	rndX = std::bind(x, engine0);
	rndY = std::bind(y, engine1);
	rndZ = std::bind(z, engine2);
}

void BoidScene::RenderScene()
{
	Scene::RenderScene();
	for (unsigned int i = 0; i < boids.size(); ++i)
		boids[i]->OnRenderObject();

}

#if !CUDA
#if THREADED
void BoidScene::UpdateScene(float dt)
{
	count += dt;
	if (count > 2500.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	size_t distribution = boids.size() / NUMBER_OF_THREADS;
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 0, distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, distribution + 1, 2 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 2 * distribution + 1, 3 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 3 * distribution + 1, 4 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 4 * distribution + 1, 5 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 5 * distribution + 1, 6 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 6 * distribution + 1, 7 * distribution, dt));
	futures.push_back(std::async(std::launch::async, &BoidScene::UpdatePartition, this, 7 * distribution + 1, boids.size() - 1, dt));

	for (auto& future : futures)
	{
		future.get();
	}
	futures.clear();

	Scene::UpdateScene(dt);
}

void BoidScene::UpdatePartition(size_t begin, size_t end, float dt)
{
	glm::vec3 posA, posB;
	for (unsigned int i = begin; i <= end; ++i)
	{
		posA = boids[i]->GetPosition();
		for (unsigned int j = 0; j < boids.size(); ++j)
		{
			if (i != j)
			{
				posB = boids[j]->GetPosition();
				float dist = glm::length(posA - posB);
				if (dist <= MAX_DISTANCE)
				{
					BoidNeighbour bNA;
					bNA.n = boids[j];
					bNA.dist = dist;
					boids[i]->AddNeighbour(bNA);
					boids[i]->UpdateFlockHeading(m_FlockHeading);
				}
			}
		}
		boids[i]->OnUpdateObject(dt);
	}
}
#else
void BoidScene::UpdateScene(float dt)
{
	count += dt;
	if (count > 2000.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	glm::vec3 posA, posB;
	for (unsigned int i = 0; i < boids.size() - 1; ++i)
	{
		posA = boids[i]->GetPosition();
		for (unsigned int j = i + 1; j < boids.size(); ++j)
		{
			if (i != j)
			{
				posB = boids[j]->GetPosition();
				float dist = glm::length(posA - posB);
				BoidNeighbour bNA;
				bNA.n = boids[j];
				bNA.dist = dist;
				boids[i]->AddNeighbour(bNA);

				BoidNeighbour bNB;
				bNB.n = boids[i];
				bNB.dist = dist;
				boids[j]->AddNeighbour(bNB);
			}
		}
	}

	for (unsigned int i = 0; i < boids.size(); ++i)
		boids[i]->OnUpdateObject(dt);

	Scene::UpdateScene(dt);
}
#endif
#else
void BoidScene::UpdateScene(float dt)
{
	count += dt;
	if (count > 2500.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}
	compute_KNN << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsGPU, maxBoids, MAX_DISTANCE, dt, m_FlockHeading);
	cudaDeviceSynchronize();

	for (unsigned int i = 0; i < maxBoids; ++i)
	{
		boids[i]->SetWorldTransform(boidsGPU[i].m_WorldTransform);
	}
	Scene::UpdateScene(dt);
}

__global__ void compute_KNN(BoidGPU* boid, const uint32_t maxBoids, const float MAX_DISTANCE, float dt, glm::vec3 heading)
{
	int tid_x = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid_x >= maxBoids)
		return;

	BoidGPU& temp = boid[tid_x];
	temp.lastIndex = 0;
	unsigned int counter = 0;

	for (unsigned int i = 0; i < maxBoids; ++i)
	{
		if (glm::distance(temp.m_Position, boid[i].m_Position) <= MAX_DISTANCE)
		{
			if (counter < MAX_NEIGHBOURS)
			{
				temp.neighbours[counter++] = &boid[i];
			}
			else
			{
				temp.lastIndex = counter;
				break;
			}
		}
	}		
	//temp.lastIndex = counter;
	updateBoid(temp, dt, maxBoids, heading);
}

__device__ void CalcCohesion(BoidGPU& boid, glm::vec3& cohVec)
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		avgPos += boid.neighbours[i]->m_Position;
	}

	avgPos /= float(boid.lastIndex - 1);
	cohVec = (avgPos - boid.m_Position);

	float mag = glm::length(cohVec);
	glm::normalize(cohVec);

	cohVec *= (0.3f * (mag * 0.001f));
	cohVec -= boid.m_Velocity;
}

__device__ void CalcSeperation(BoidGPU& boid, glm::vec3& sepVec)
{
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		glm::vec3 dir = boid.neighbours[i]->m_Position - boid.m_Position;
		float len = sqrtf(glm::dot(dir, dir));
		sepVec -= (dir / len);
	}
	sepVec /= float(boid.lastIndex - 1);
	sepVec *= 0.25f;
}

__device__ void CalcAlignment(BoidGPU& boid, glm::vec3& alignVec)
{
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		avgVel += boid.neighbours[i]->m_Velocity;
	}
	avgVel /= float(boid.lastIndex - 1);
	alignVec = (avgVel - boid.m_Velocity);
}

__device__ void LimitVelocity(BoidGPU& boid)
{
	float speed = sqrtf(glm::dot(boid.m_Velocity, boid.m_Velocity));
	if (speed > 0.3f)
	{
		boid.m_Velocity = (boid.m_Velocity / speed);
	}
	boid.m_Velocity *= 0.999f;
}

__device__ void updateBoid(BoidGPU& boid, float dt, const uint32_t maxBoids, glm::vec3& heading)
{
	glm::vec3 cohVec(0, 0, 0);
	glm::vec3 sepVec(0, 0, 0);
	glm::vec3 alignVec(0, 0, 0);
	CalcCohesion(boid, cohVec);
	CalcSeperation(boid, sepVec);
	CalcAlignment(boid, alignVec);

	boid.m_Velocity = cohVec + sepVec + alignVec + ((heading - boid.m_Position) * 0.001f);
	LimitVelocity(boid);
	boid.m_OldPosition = boid.m_Position;
	boid.m_Position += (boid.m_Velocity * dt);
	boid.m_WorldTransform = glm::translate(glm::mat4(1.0f), boid.m_OldPosition + boid.m_Position);
}
#endif