//Author: Shaun Heald
#include "BoidScene.h"
#include "Common.h"
#include <iostream>
#if CUDA
#include "OGLRenderer.h"
#include <cuda_gl_interop.h>

struct cudaGraphicsResource* modelMatricesCUDA;
#endif

//k value
#define MAX_DISTANCE 100.0f
#define MAX_DISTANCE_SQR MAX_DISTANCE * MAX_DISTANCE
#define MAX_SPEED 4.0f
#define MAX_SPEED_SQR MAX_SPEED * MAX_SPEED
#define DIV 1.0f / (THREADS_PER_BLOCK - 1.0f)

BoidScene::BoidScene(unsigned int numberOfBoids, Shader* shader, Mesh* mesh)
{
	numBoids = numberOfBoids;
#if CUDA
	cudaSetDevice(0);
	//Enable OpenGL interop
	cudaGraphicsGLRegisterBuffer(&modelMatricesCUDA, OGLRenderer::Instance()->GetSSBO_ID(), cudaGraphicsMapFlagsWriteDiscard);
	//Ensure one thread per Boid
	BLOCKS_PER_GRID = (numBoids + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;
	//Allocate device memory, using unified memory to de-reference pointer from host side
	cudaMallocManaged((void**)&boidsDevice, sizeof(BoidGPU));
	cudaMallocManaged((void**)&boidsDevice->m_Position, sizeof(glm::vec3) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_Velocity, sizeof(glm::vec3) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_AlignmentVector, sizeof(glm::vec3) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_SeperationVector, sizeof(glm::vec3) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_CohesiveVector, sizeof(glm::vec3) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_Key, sizeof(int) * numBoids);
	cudaMallocManaged((void**)&boidsDevice->m_Val, sizeof(unsigned int) * numBoids);	
	//This is only used for GPU #3 implementation - gets device pointer
	dev_key_ptr = thrust::device_pointer_cast(boidsDevice->m_Key);
	dev_val_ptr = thrust::device_pointer_cast(boidsDevice->m_Val);	
	//Debug timing stuff
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
#endif

	InitGenerator(numBoids);

	glm::vec3 pos, vel;
	for (unsigned int i = 0; i < numBoids; ++i)
	{
		pos = glm::vec3(rndX(), rndY(), rndZ());
		vel = glm::vec3(rndX(), rndY(), rndZ());

		Boid* b = new Boid(pos, vel);
		b->SetRenderComponent(new RenderComponent(mesh, shader));

#if CUDA
		//Copy data to device - unified memory so this is ok
		boidsDevice->m_Position[i] = pos;
		boidsDevice->m_Velocity[i] = vel;
#endif
		boids.push_back(b);
	}
	m_FlockHeading = glm::vec3(0, 0, 0);
#if THREADED
	futures.clear();
#endif
}

BoidScene::~BoidScene()
{
	//Cleanup memory
#if CUDA
	cudaDeviceSynchronize();
	cudaFree(boidsDevice->m_AlignmentVector);
	cudaFree(boidsDevice->m_CohesiveVector);
	cudaFree(boidsDevice->m_Key);
	cudaFree(boidsDevice->m_Position);
	cudaFree(boidsDevice->m_SeperationVector);
	cudaFree(boidsDevice->m_Val);
	cudaFree(boidsDevice->m_Velocity);
	cudaFree(boidsDevice);
	cudaGraphicsUnregisterResource(modelMatricesCUDA);
	cudaDeviceReset();
#endif
	for (auto& b : boids)
	{
		delete b;
		b = nullptr;
	}
	boids.clear();
}
//Randomly generates points to place Boids.
void BoidScene::InitGenerator(int spread)
{
	std::random_device rD0;
	std::random_device rD1;
	std::random_device rD2;
	std::default_random_engine engine0(rD0());
	std::default_random_engine engine1(rD1());
	std::default_random_engine engine2(rD2());

	std::uniform_real_distribution<float> x(-200.0f, 200.0f);
	std::uniform_real_distribution<float> y(-200.0f, 200.0f);
	std::uniform_real_distribution<float> z(-200.0f, 200.0f);

	rndX = std::bind(x, engine0);
	rndY = std::bind(y, engine1);
	rndZ = std::bind(z, engine2);
}

void BoidScene::RenderScene()
{
	Scene::RenderScene(); 
#if CUDA
	boids[0]->OnRenderObject();
#else
	for (unsigned int i = 0; i < boids.size(); ++i)
		boids[i]->OnRenderObject();
#endif
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
	//Split data over 8 threads
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
//Each thread runs this function and works on a subset of data
void BoidScene::UpdatePartition(size_t begin, size_t end, float dt)
{
	glm::vec3 posA, posB;
	glm::vec3 dir;
	for (unsigned int i = begin; i <= end; ++i)
	{
		posA = boids[i]->GetPosition();
		for (unsigned int j = 0; j < boids.size(); ++j)
		{
			if (i != j)
			{
				posB = boids[j]->GetPosition();
				dir = posB - posA;
				float dist = glm::dot(dir, dir);
				BoidNeighbour bNA;
				bNA.n = boids[j];
				bNA.dist = dist ;
				boids[i]->AddNeighbour(bNA);
			}
		}
		boids[i]->OnUpdateObject(dt);
	}
}
#else
//Single threaded version.
void BoidScene::UpdateScene(float dt)
{
	count += dt;
	if (count > 2000.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	glm::vec3 posA, posB;
	glm::vec3 dir;
	float distance = 0.0f;
	for (unsigned int i = 0; i < boids.size() - 1; ++i)
	{
		posA = boids[i]->GetPosition();
		for (unsigned int j = i + 1; j < boids.size(); ++j)
		{
			if (i != j)
			{
				dir = posB - posA;
				distance = glm::dot(dir, dir);
				posB = boids[j]->GetPosition();
				BoidNeighbour bNA;
				bNA.n = boids[j];
				bNA.dist = distance;
				boids[i]->AddNeighbour(bNA);

				BoidNeighbour bNB;
				bNB.n = boids[i];
				bNB.dist = distance;
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
	float temp = 0.0f;
	cudaEventRecord(start, 0);
	//Map ogl buffer for use with CUDA
	cudaGraphicsMapResources(1, &modelMatricesCUDA, 0);
	//Compute the nearest neighbours
	ComputeKNN << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice, numBoids);
	
#if KERNEL == 2
	//Parallel radix sort
	thrust::sort_by_key(dev_key_ptr, dev_key_ptr + numBoids, dev_val_ptr);
	ComputeRules << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice);
#endif
	//Get ogl buffer pointer
	cudaGraphicsResourceGetMappedPointer((void**)&modelMatricesDevice, nullptr, modelMatricesCUDA);
	UpdateBoid << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(boidsDevice, modelMatricesDevice, m_FlockHeading, dt);
	//Release ogl buffer 
	cudaGraphicsUnmapResources(1, &modelMatricesCUDA, 0);
	//Debug timing stuff
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&temp, start, stop);
	elapsed_cuda += temp;

	count += dt;
	if (count > 500.0f)
	{
		m_FlockHeading = glm::vec3(rndX(), rndY(), rndZ());
		count = 0.0f;
	}

	Scene::UpdateScene(dt);
}
#if KERNEL == 0
//Slow version - N squared
__global__ void ComputeKNN(BoidGPU* boids, unsigned int numBoids)
{
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	glm::vec3 myPos = boids->m_Position[tid];
	glm::vec3 myVel = boids->m_Velocity[tid];
	glm::vec3 temp_cohVec = glm::vec3(0, 0, 0);
	glm::vec3 temp_sepVec = glm::vec3(0, 0, 0);
	glm::vec3 temp_alignVec = glm::vec3(0, 0, 0);	
	glm::vec3 dir = glm::vec3(0,0,0);
	float distance = 0.0f;

#pragma unroll	
	for (unsigned int i = 0; i < numBoids; ++i)
	{
		dir = boids->m_Position[i] - myPos;
		//(dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z)
		distance = __fadd_rn(__fadd_rn(__fmul_rn(dir.x, dir.x), __fmul_rn(dir.y, dir.y)), __fmul_rn(dir.z, dir.z));

		if (distance <= MAX_DISTANCE_SQR)
		{
			distance = __frsqrt_rn(distance + 0.001f);
			temp_cohVec += boids->m_Position[i];
			temp_sepVec -= dir * distance;
			temp_alignVec += boids->m_Velocity[i];
		}
	}	

	boids->m_CohesiveVector[tid] = (temp_cohVec - myPos) * 0.01f;
	boids->m_SeperationVector[tid] = temp_sepVec;
	boids->m_AlignmentVector[tid] = (temp_alignVec - myVel) * 0.8f;
}
#endif

#if KERNEL == 1
//Medium version - using shared memory to optimise kernel - compute in 'tiles'.
__global__ void ComputeKNN(BoidGPU* boids, unsigned int numBoids)
{
	__shared__ glm::vec3 shPos[THREADS_PER_BLOCK];
	__shared__ glm::vec3 shVel[THREADS_PER_BLOCK];
	
	int gid = threadIdx.x + (blockIdx.x * blockDim.x);

	int idx;
	float distance = 0.0f;
	glm::vec3 dir(0, 0, 0);
	glm::vec3 myPos = boids->m_Position[gid];
	glm::vec3 myVel = boids->m_Velocity[gid];
	glm::vec3 temp_cohVec(0, 0, 0);
	glm::vec3 temp_sepVec(0, 0, 0);
	glm::vec3 temp_alignVec(0, 0, 0);

#pragma unroll
	for (int i = 0, tile = 0; i < numBoids; i += THREADS_PER_BLOCK, ++tile)
	{
		idx = tile * THREADS_PER_BLOCK + threadIdx.x;
		shPos[threadIdx.x] = boids->m_Position[idx];
		shVel[threadIdx.x] = boids->m_Velocity[idx];
		__syncthreads();

#pragma unroll
		for (int j = 0; j < THREADS_PER_BLOCK; ++j)
		{
			dir = shPos[j] - myPos;
			//(dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z)
			distance = __fadd_rn(__fadd_rn(__fmul_rn(dir.x, dir.x), __fmul_rn(dir.y, dir.y)), __fmul_rn(dir.z, dir.z));

			if (distance <= MAX_DISTANCE_SQR)
			{
				distance = __frsqrt_rn(distance + 0.001f);
				temp_cohVec += shPos[j];	
				temp_sepVec -= dir * distance;
				temp_alignVec += shVel[j];
			}
		}
		__syncthreads();
	}

	boids->m_CohesiveVector[gid] = (temp_cohVec - myPos) * 0.01f;
	boids->m_SeperationVector[gid] = temp_sepVec;
	boids->m_AlignmentVector[gid] = (temp_alignVec - myVel) * 0.8f;
}
#endif

#if KERNEL == 2
//Fast version - hash position, sort using thrust library and compute rules.
__global__ void ComputeKNN(BoidGPU* boids, unsigned int numBoids)
{
	unsigned int gid = threadIdx.x + (blockIdx.x * blockDim.x);

	glm::vec3 temp_pos = boids->m_Position[gid];
	//unsigned int(((temp_pos.x + temp_pos.y + temp_pos.z) * 13) * 17);
	int hash = __float2int_rn(__fmul_rn(__fmul_rn(__fadd_rn(__fadd_rn(temp_pos.x, temp_pos.y), temp_pos.z), 13), 17));
	boids->m_Key[gid] = hash;
	boids->m_Val[gid] = gid;
}

__global__ void ComputeRules(BoidGPU* boids)
{
	__shared__ glm::vec3 shPos[THREADS_PER_BLOCK];
	__shared__ glm::vec3 shVel[THREADS_PER_BLOCK];

	unsigned int gid = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int sortedID = boids->m_Val[gid];
	glm::vec3 myPos = shPos[threadIdx.x] = boids->m_Position[sortedID];
	glm::vec3 myVel = shVel[threadIdx.x] = boids->m_Velocity[sortedID];

	__syncthreads();

	float distance = 0.0f;
	glm::vec3 dir(0, 0, 0);
	glm::vec3 temp_cohVec(0, 0, 0);
	glm::vec3 temp_sepVec(0, 0, 0);
	glm::vec3 temp_alignVec(0, 0, 0);

#pragma unroll
	for (unsigned int i = 0; i < THREADS_PER_BLOCK; ++i)
	{
		dir = shPos[i] - myPos;
		//1.0f / sqrtf((dir.x * dir.x) + (dir.y * dir.y) + (dir.z * dir.z) + 0.001f)
		distance = __frsqrt_rn(__fadd_rn(__fadd_rn(__fmul_rn(dir.x, dir.x), __fmul_rn(dir.y, dir.y)), __fmul_rn(dir.z, dir.z)) + 0.001f);

		temp_cohVec += shPos[i];
		temp_sepVec -= dir * distance;
		temp_alignVec += shVel[i];
	}

	boids->m_CohesiveVector[sortedID] = ((temp_cohVec) - myPos) * 0.01f;
	boids->m_SeperationVector[sortedID] = temp_sepVec;
	boids->m_AlignmentVector[sortedID] = ((temp_alignVec) - myVel) * 0.8f;
}
#endif
//Update velocity and position and calculate model matrix then store it to ogl shared buffer
__global__ void UpdateBoid(BoidGPU* boids, glm::mat4* boidMat, const glm::vec3 heading, const float dt)
{
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);

	glm::vec3 velocity = glm::vec3(boids->m_Velocity[tid]);
	velocity += (boids->m_CohesiveVector[tid] + boids->m_SeperationVector[tid] + boids->m_AlignmentVector[tid] 
		+ (glm::cross(heading, boids->m_Position[tid]) * 0.05f)) * __frcp_rn(dt);
	
	//(((velocity.x * velocity.x) + (velocity.y * velocity.y) + (velocity.z * velocity.z)));
	float speed = __fadd_rn(__fadd_rn(__fmul_rn(velocity.x, velocity.x), __fmul_rn(velocity.y, velocity.y)), __fmul_rn(velocity.z, velocity.z));
	
	if (speed > MAX_SPEED_SQR)
	{
		//1.0f / sqrt(x)
		velocity = (velocity * __frsqrt_rn(speed)) * MAX_SPEED;
	}

	velocity *= 0.999f;
	boids->m_Velocity[tid] = velocity;
	boids->m_Position[tid] += velocity * __frcp_rn(dt);

	boidMat[tid] = glm::mat4_cast(glm::quat(velocity)) * glm::translate(boids->m_Position[tid]);
}
#endif