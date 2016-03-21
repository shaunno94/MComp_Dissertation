#include "Common.h"
#if CUDA
#include "OGLRenderer.h"
#include "BoidScene.h"
#include "BoidGeneratorGPU.h"
#include "Timer.h"
#include "Shader.h"
#include "Mesh.h"

const uint32_t NUM_BOIDS = 8192;
/*const uint32_t THREADS_K1 = 32;
const uint32_t THREADS_K2 = 1024;
const uint32_t BLOCKS_K1 = (NUM_BOIDS + (THREADS_K1 - 1)) / THREADS_K1;
const uint32_t BLOCKS_K2 = (NUM_BOIDS + (THREADS_K2 - 1)) / THREADS_K2;
const dim3 THREAD_DIM_K1 = dim3(THREADS_K1, THREADS_K1);
const dim3 BLOCK_DIM_K1 = dim3(BLOCKS_K1, BLOCKS_K1);*/

//while (tid < NUM_BOIDS)//tid += blockDim.x * gridDim.x;		//int offset = x + y * blockDim.x * gridDim.x;
//cudaMalloc((void**)&dev_a, NUM_BOIDS * sizeof(BoidGPU));//cudaMemcpy(dev_a, boidGPU->GetBoidData().data(), NUM_BOIDS * sizeof(BoidGPU), cudaMemcpyHostToDevice);

/*__global__ void compute_KNN(BoidGPU* boid)
{	
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

	if (glm::distance(boid[tid_x].m_Position, boid[tid_y].m_Position) <= MAX_DIST)
	{	
		if (boid[tid_x].lastIndex < 50)
		{
			unsigned int index = atomicAdd(&boid[tid_x].lastIndex, 1);
			boid[tid_x].neighbours[index] = &boid[tid_y];
		}
	}	
}

__global__ void compute_KNN(BoidGPU* boid)
{
	int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid_x >= NUM_BOIDS)
		return;

	BoidGPU& temp = boid[tid_x];
	temp.lastIndex = 0;
	unsigned int counter = 0;

	for (unsigned int i = 0; i < NUM_BOIDS; ++i)
	{
		if (glm::distance(temp.m_Position, boid[i].m_Position) <= MAX_DIST)
		{
			if (counter < 50)
			{
				temp.neighbours[counter] = &boid[i];
				counter++;
			}
			else
			{
				temp.lastIndex = counter;
				return;
			}
		}
	}
}

__device__ void CalcCohesion(BoidGPU& boid, glm::vec3& cohVec)
{
	glm::vec3 avgPos = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		avgPos += boid.neighbours[i]->m_Position;
	}

	avgPos /= 50.0f;
	cohVec = (avgPos - boid.m_Position);

	float mag = glm::length(cohVec);
	glm::normalize(cohVec);

	cohVec *= (0.25f * (mag * 0.001f));
	cohVec -= boid.m_Velocity;
}

__device__ void CalcSeperation(BoidGPU& boid, glm::vec3& sepVec)
{
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		sepVec -= (boid.neighbours[i]->m_Position - boid.m_Position) / glm::distance(boid.neighbours[i]->m_Position, boid.m_Position);
	}
	sepVec /= 50.0f;
	sepVec *= 0.3f;
}

__device__ void CalcAlignment(BoidGPU& boid, glm::vec3& alignVec)
{
	glm::vec3 avgVel = glm::vec3(0, 0, 0);
	for (unsigned int i = 0; i < boid.lastIndex; ++i)
	{
		avgVel += boid.neighbours[i]->m_Velocity;
	}
	avgVel /= 50.0f;
	alignVec = (avgVel - boid.m_Velocity) * 0.8f;
}

__global__ void updateBoids(BoidGPU* boid, float dt)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= NUM_BOIDS)
		return;
	
	BoidGPU& temp = boid[tid];
	glm::vec3 cohVec(0, 0, 0);
	glm::vec3 sepVec(0, 0, 0);
	glm::vec3 alignVec(0, 0, 0);
	CalcCohesion(temp, cohVec);
	CalcSeperation(temp, sepVec);
	CalcAlignment(temp, alignVec);

	temp.m_Velocity = cohVec + sepVec + alignVec;
	temp.m_Position += (temp.m_Velocity * dt);
	temp.m_WorldTransform = glm::translate(glm::mat4(1.0f), temp.m_Position);
	temp.lastIndex = 0;
}*/

int main(void)
{
	OGLRenderer* renderer = OGLRenderer::Instance();
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle();
	BoidScene* boidScene = new BoidScene(NUM_BOIDS, simpleShader, triMesh);
	//BoidGeneratorGPU boidGPU(NUM_BOIDS);
	Timer gt;

	/*compute_KNN << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData());
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;*/

	/*updateBoids << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData(), 16.0f);
	cudaDeviceSynchronize();
	std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;*/

	/*for (unsigned int i = 0; i < NUM_BOIDS; ++i)
	{
		std::cout << boidGPU.GetBoidData()[i].lastIndex << std::endl;
	}*/

	renderer->SetCurrentScene(boidScene);
	//Main loop.
	while (renderer->ShouldClose()) //Check if the ESC key was pressed or the window was closed
	{
		gt.startTimer();
		//compute_KNN << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData());
		//updateBoids << <BLOCKS_K2, THREADS_K2 >> >(boidGPU.GetBoidData(), 16.0f);
		//std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
		
		renderer->Render(gt.getLast());
		glfwPollEvents();
		gt.stopTimer();
	}

	OGLRenderer::Release();
	delete triMesh;
	delete simpleShader;
	cudaDeviceReset();
	return 0;
}
#endif