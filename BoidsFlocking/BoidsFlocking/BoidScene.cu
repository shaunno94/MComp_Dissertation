#include "BoidScene.h"
#include "Common.h"
#include <iostream>

//k value
#define MAX_DISTANCE 35.0f
#define MAX_DISTANCE_SQR MAX_DISTANCE * MAX_DISTANCE
#define MAX_SPEED 0.25f
#define MAX_SPEED_SQR MAX_SPEED / MAX_SPEED

BoidScene::BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh)
{
	InitGenerator(numberOfBoids);
#if CUDA
	cudaSetDevice(0);
	BLOCKS_PER_GRID = (numberOfBoids + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	m_Position = (glm::vec3*)malloc(numberOfBoids * sizeof(glm::vec3));
	m_Velocity = (glm::vec3*)malloc(numberOfBoids * sizeof(glm::vec3));
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
		m_Position[i] = pos;
		m_Velocity[i] = vel;
		b->SetWorldTransform(&modelMatricesHostPinned[i]);
#endif
		boids.push_back(b);
	}
	m_FlockHeading = glm::vec3(0, 0, 0);

#if CUDA
	cudaMalloc((void**)&boidsDevice, sizeof(BoidGPU));
	cudaMalloc((void**)&modelMatricesDevice, numberOfBoids * sizeof(glm::mat4));
	cudaMemcpy(boidsDevice->m_Position, m_Position, numberOfBoids * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(boidsDevice->m_Velocity, m_Velocity, numberOfBoids * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	dev_key_ptr = thrust::device_pointer_cast(boidsDevice->m_Key);
	dev_val_ptr = thrust::device_pointer_cast(boidsDevice->m_Val);
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
	cudaDeviceReset();
	free(m_Position);
	free(m_Velocity);
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
	thrust::sort_by_key(dev_key_ptr, dev_key_ptr + NUM_BOIDS, dev_val_ptr);
	ComputeRules << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
	UpdateBoid << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice, modelMatricesDevice, m_FlockHeading, dt);
	cudaMemcpyAsync(modelMatricesHostPinned, modelMatricesDevice, sizeof(glm::mat4) * NUM_BOIDS, cudaMemcpyDeviceToHost);
	
	count += dt;
	if (count > 2500.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	Scene::UpdateScene(dt);
}

/*__global__ void ComputeKNN(BoidGPU* boids)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 myPos = boids->m_Position[tid];
	glm::vec3 myVel = boids->m_Velocity[tid];
	glm::vec3 temp_cohVec = glm::vec3(0, 0, 0);
	glm::vec3 temp_sepVec = glm::vec3(0, 0, 0);
	glm::vec3 temp_alignVec = glm::vec3(0, 0, 0);	
	glm::vec3 dir = glm::vec3(0,0,0);
	float counter = 0;
	float dist = 0.0f;

#pragma unroll	
	for (unsigned int i = 0; i < NUM_BOIDS; ++i)
	{
		if (tid == i) continue;

		dir = boids->m_Position[i] - myPos;
		dist = glm::dot(dir, dir);

		if (dist <= MAX_DISTANCE_SQR)
		{
			counter += 1.0f;
			temp_cohVec += boids->m_Position[i];
			temp_sepVec -= dir * (1.0f / sqrtf(dist));
			temp_alignVec += boids->m_Velocity[i];
		}
	}	

	boids->m_CohesiveVector[tid] = (((temp_cohVec / counter) - myPos) * 0.0001f) - myVel;
	boids->m_SeperationVector[tid] = (temp_sepVec / counter) * 0.25f;
	boids->m_AlignmentVector[tid] = ((temp_alignVec / counter) - myVel);
}*/

/*__global__ void ComputeKNN(BoidGPU* boids)
{
	__shared__ glm::vec3 shPos[THREADS_PER_BLOCK];
	__shared__ glm::vec3 shVel[THREADS_PER_BLOCK];
	
	int gid = threadIdx.x + (blockIdx.x * blockDim.x);

	int idx = threadIdx.x;
	float counter = 0.0f;
	float distSQR = 0.0f;
	glm::vec3 dir(0, 0, 0);
	glm::vec3 myPos = boids->m_Position[gid];
	glm::vec3 myVel = boids->m_Velocity[gid];
	glm::vec3 temp_cohVec(0, 0, 0);
	glm::vec3 temp_sepVec(0, 0, 0);
	glm::vec3 temp_alignVec(0, 0, 0);

#pragma unroll
	for (int i = 0; i < NUM_BOIDS; i += THREADS_PER_BLOCK)
	{
		//idx = tile * THREADS_PER_BLOCK + threadIdx.x;
		shPos[threadIdx.x] = boids->m_Position[idx];
		shVel[threadIdx.x] = boids->m_Velocity[idx];
		idx += THREADS_PER_BLOCK;
		__syncthreads();

#pragma unroll
		for (int j = 0; j < THREADS_PER_BLOCK; ++j)
		{
			//if (j == threadIdx.x) continue;

			dir = shPos[j] - myPos;
			distSQR = (dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z);
			if (distSQR == 0.0f) continue;

			if (distSQR <= MAX_DISTANCE_SQR)
			{
				counter += 1.0f;
				temp_cohVec += shPos[j];	
				temp_sepVec -= dir * (1.0f / sqrtf(distSQR));
				temp_alignVec += shVel[j];
			}
		}
		__syncthreads();
	}
	counter = 1.0f / counter;
	boids->m_CohesiveVector[gid] = (((temp_cohVec * counter) - myPos) * 0.001f) - myVel;
	boids->m_SeperationVector[gid] = (temp_sepVec * counter) * 0.25f;
	boids->m_AlignmentVector[gid] = ((temp_alignVec * counter) - myVel) * 0.8f;
}*/

__global__ void ComputeKNN(BoidGPU* boids)
{
	int gid = threadIdx.x + (blockIdx.x * blockDim.x);

	glm::vec3 temp_pos = boids->m_Position[gid];
	unsigned int hash = unsigned int(((temp_pos.x + temp_pos.y + temp_pos.z) * 13) * 17) % 144;
	boids->m_Key[gid] = hash;
	boids->m_Val[gid] = gid;
}

__global__ void ComputeRules(BoidGPU* boids)
{
	__shared__ glm::vec3 shPos[THREADS_PER_BLOCK];
	__shared__ glm::vec3 shVel[THREADS_PER_BLOCK];

	int gid = threadIdx.x + (blockIdx.x * blockDim.x);
	int sortedID = boids->m_Val[gid];
	shPos[threadIdx.x] = boids->m_Position[sortedID];
	shVel[threadIdx.x] = boids->m_Velocity[sortedID];

	__syncthreads();

	float counter = 0.0f;
	float distSQR = 0.0f;
	glm::vec3 dir(0, 0, 0);
	glm::vec3 myPos = boids->m_Position[gid];
	glm::vec3 myVel = boids->m_Velocity[gid];
	glm::vec3 temp_cohVec(0, 0, 0);
	glm::vec3 temp_sepVec(0, 0, 0);
	glm::vec3 temp_alignVec(0, 0, 0);

#pragma unroll
	for (int i = 0; i < THREADS_PER_BLOCK; ++i)
	{
		dir = shPos[i] - myPos;
		distSQR = (dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z);
		if (distSQR == 0.0f) continue;

		counter += 1.0f;
		temp_cohVec += shPos[i];
		temp_sepVec -= dir * (1.0f / sqrtf(distSQR));
		temp_alignVec += shVel[i];
	}

	counter = 1.0f / counter;
	boids->m_CohesiveVector[gid] = (((temp_cohVec * counter) - myPos) * 0.001f) - myVel;
	boids->m_SeperationVector[gid] = (temp_sepVec * counter) * 0.25f;
	boids->m_AlignmentVector[gid] = ((temp_alignVec * counter) - myVel) * 0.8f;
}

__global__ void UpdateBoid(BoidGPU* boids, glm::mat4* boidMat, const glm::vec3 heading, const float dt)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= NUM_BOIDS)
		return;

	glm::vec3 velocity = boids->m_Velocity[tid];
	velocity += ((boids->m_CohesiveVector[tid] + boids->m_SeperationVector[tid] + boids->m_AlignmentVector[tid]) + ((heading - boids->m_Position[tid]) * 0.001f)) * dt;
	float speed = sqrtf(((velocity.x * velocity.x) + (velocity.y * velocity.y) + (velocity.z * velocity.z)));
	
	if (speed > MAX_SPEED)
	{
		//speed = sqrtf(speed);
		velocity = (velocity * (1.0f / speed)) * MAX_SPEED;
	}

	boids->m_Velocity[tid] = velocity * 0.999f;
	boids->m_Position[tid] += (velocity * dt);

	velocity = glm::normalize(velocity);
	boidMat[tid] = glm::mat4_cast(glm::quat(velocity)) * glm::translate(boids->m_Position[tid]);
}
#endif