#include "BoidScene.h"
#include "Common.h"
#include <iostream>

#if CUDA
#define MAX_NEIGHBOURS 150
#define THREADS_PER_BLOCK 256
#endif
//k value
#define MAX_DISTANCE 65.0f
#define MAX_DISTANCE_SQR MAX_DISTANCE * MAX_DISTANCE

BoidScene::BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh)
{
	InitGenerator(numberOfBoids);
#if CUDA
	cudaSetDevice(0);
	BLOCKS_PER_GRID = (numberOfBoids + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	boidsHost = (BoidGPU*)malloc(numberOfBoids * sizeof(BoidGPU));
	cudaMallocHost((void**)&modelMatricesHostPinned, numberOfBoids * sizeof(glm::mat4));	
#endif

	glm::vec3 pos, vel;
	for (unsigned int i = 0; i < numberOfBoids; ++i)
	{
		pos = glm::vec3(rndX(), rndY(), rndZ());
		vel = glm::vec3(rndX(), rndY(), rndZ());

		Boid* b = new Boid(pos, vel);
		b->SetRenderComponent(new RenderComponent(mesh, shader));

#if CUDA
		boidsHost[i] = BoidGPU(MAX_NEIGHBOURS, pos, vel);
		b->SetWorldTransform(&modelMatricesHostPinned[i]);
#endif
		boids.push_back(b);
	}
	m_FlockHeading = glm::vec3(0, 0, 0);

#if CUDA
	cudaMalloc((void**)&boidsDevice, numberOfBoids * sizeof(BoidGPU));
	cudaMalloc((void**)&modelMatricesDevice, numberOfBoids * sizeof(glm::mat4));
	cudaMemcpyAsync(boidsDevice, boidsHost, numberOfBoids * sizeof(BoidGPU), cudaMemcpyHostToDevice);
	//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif

#if THREADED
	futures.clear();
#endif
}

BoidScene::~BoidScene()
{
#if CUDA
	cudaDeviceSynchronize();
	cudaFree(boidsDevice);
	cudaFree(modelMatricesDevice);
	cudaFree(modelMatricesHostPinned);
	cudaDeviceReset();
	free(boidsHost);
#endif
	for (auto& b : boids)
	{
		delete b;
		b = nullptr;
	}
	boids.clear();
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
		Boid::UpdateFlockHeading(m_FlockHeading);
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
				float dist = glm::distance(posA, posB);
				if (dist <= MAX_DISTANCE)
				{
					BoidNeighbour bNA;
					bNA.n = boids[j];
					bNA.dist = dist;
					boids[i]->AddNeighbour(bNA);
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
	ComputeKNN << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
	//CalcCohesion << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
	//CalcAlignment << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
	//CalcSeperation << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
	CalcVelocity << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice, m_FlockHeading);
	UpdateBoid << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice, modelMatricesDevice, dt);
	cudaMemcpyAsync(modelMatricesHostPinned, modelMatricesDevice, sizeof(glm::mat4) * NUM_BOIDS, cudaMemcpyDeviceToHost);
	
	count += dt;
	if (count > 2500.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	Scene::UpdateScene(dt);
}

/*__global__ void ComputeKNN(BoidGPU* boid)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 temp_pos = boid[tid].m_Position;
	Neighbour* temp_neighbour = boid[tid].neighbours;
	unsigned int counter = 0;
	float dist = 0.0f;
	glm::vec3 dir = glm::vec3(0,0,0);

#pragma unroll	
	for (unsigned int i = 0; i < NUM_BOIDS; ++i)
	{
		if (tid == i) continue;

		dir = boid[i].m_Position - temp_pos;
		dist = glm::dot(dir, dir);
		if (dist <= MAX_DISTANCE_SQR)
		{
			temp_neighbour[counter].n = i;
			temp_neighbour[counter++].distance = 1.0f / sqrtf(dist);
			if (counter >= MAX_NEIGHBOURS) break;
		}
	}	

	boid[tid].neighbours = temp_neighbour;
	boid[tid].lastIndex = counter;
}*/

__global__ void ComputeKNN(BoidGPU* boid)
{
	__shared__ glm::vec3 shPos[THREADS_PER_BLOCK];
	__shared__ glm::vec3 shVel[THREADS_PER_BLOCK];
	
	int gid = threadIdx.x + (blockIdx.x * blockDim.x);

	//if (tid < NUM_BOIDS)
	{
		int idx;
		float counter = 0.0f;
		float dist = 0.0f;
		glm::vec3 dir = glm::vec3(0, 0, 0);
		glm::vec3 myPos = boid[gid].m_Position;
		glm::vec3 myVel = boid[gid].m_Velocity;
		glm::vec3 temp_cohVec = glm::vec3(0, 0, 0);
		glm::vec3 temp_sepVec = glm::vec3(0, 0, 0);
		glm::vec3 temp_alignVec = glm::vec3(0, 0, 0);
#pragma unroll
		for (int i = 0, tile = 0; i < NUM_BOIDS; i += THREADS_PER_BLOCK, tile++)
		{
			idx = tile * blockDim.x + threadIdx.x;
			shPos[threadIdx.x] = boid[idx].m_Position;
			shVel[threadIdx.x] = boid[idx].m_Velocity;
			__syncthreads();

#pragma unroll
			for (int j = 0; j < blockDim.x; ++j)
			{
				dir = shPos[j] - myPos;
				dist = glm::dot(dir, dir);
				if (dist <= MAX_DISTANCE_SQR)
				{
					if (dist < 0.00001f) continue;

					counter += 1.0f;
					temp_cohVec += shPos[j];	
					temp_sepVec -= dir * (1.0f / sqrtf(dist));
					temp_alignVec += shVel[j];
				}
			}
			__syncthreads();
		}

		boid[gid].m_CohesiveVector = (((temp_cohVec / counter) - myPos) * 0.0001f) - myVel;
		boid[gid].m_SeperationVector = (temp_sepVec / counter) * 0.25f;
		boid[gid].m_AlignmentVector = ((temp_alignVec / counter) - myVel);
	}
}

__global__ void CalcCohesion(BoidGPU* boid)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 temp_cohVec = glm::vec3(0, 0, 0);
	unsigned int index = boid[tid].lastIndex;

	for (unsigned int i = 0; i < index; ++i)
	{
		temp_cohVec += boid[boid[tid].neighbours[i].n].m_Position;
	}

	temp_cohVec = glm::normalize((temp_cohVec / float(index)) - boid[tid].m_Position);
	temp_cohVec *= (0.3f * (glm::length(temp_cohVec) * 0.001f));
	boid[tid].m_CohesiveVector = (temp_cohVec - boid[tid].m_Velocity);
}

__global__ void CalcSeperation(BoidGPU* boid)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 temp_pos = boid[tid].m_Position;
	glm::vec3 temp_sepVec = glm::vec3(0, 0, 0);
	unsigned int index = boid[tid].lastIndex;

	for (unsigned int i = 0; i < index; ++i)
	{
		temp_sepVec -= ((boid[boid[tid].neighbours[i].n].m_Position - temp_pos) * boid[tid].neighbours[i].distance);
	}

	boid[tid].m_SeperationVector = (temp_sepVec / float(index)) * 0.25f;
}

__global__ void CalcAlignment(BoidGPU* boid)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	unsigned int index = boid[tid].lastIndex;
	glm::vec3 temp_alignVec = glm::vec3(0, 0, 0);

	for (unsigned int i = 0; i < index; ++i)
	{
		temp_alignVec += boid[boid[tid].neighbours[i].n].m_Velocity;
	}
	
	boid[tid].m_AlignmentVector = ((temp_alignVec / float(index)) - boid[tid].m_Velocity);
}

__global__ void CalcVelocity(BoidGPU* boid, const glm::vec3 heading)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 velocity = boid[tid].m_CohesiveVector + boid[tid].m_SeperationVector + boid[tid].m_AlignmentVector + ((heading - boid[tid].m_Position) * 0.001f);
	float speed = sqrtf(glm::dot(velocity, velocity));
	
	if (speed > 0.3f)
	{
		velocity = (velocity / speed) * 0.3f;
	}
	boid[tid].m_Velocity = velocity * 0.999f;
	velocity = glm::normalize(velocity);
	boid[tid].m_Rotation = glm::quat(glm::vec3(velocity.x, velocity.y, velocity.z));
}

__global__ void UpdateBoid(BoidGPU* boid, glm::mat4* boidMat, const float dt)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return; 	

	boid[tid].m_OldPosition = boid[tid].m_Position;
	boid[tid].m_Position += (boid[tid].m_Velocity * dt);	
	boidMat[tid] = /*glm::mat4_cast(boid[tid].m_Rotation) */ glm::translate(boid[tid].m_OldPosition + boid[tid].m_Position);
	boid[tid].lastIndex = 0;
}
#endif