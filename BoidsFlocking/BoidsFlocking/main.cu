#include "Common.h"
#if CUDA
#include "OGLRenderer.h"
#include "BoidGeneratorGPU.h"
#include "Timer.h"
#include "Shader.h"
#include "Mesh.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

const uint32_t NUM_BOIDS = 100000;
//const uint64_t RESULT_SIZE = (NUM_BOIDS * NUM_BOIDS);
const uint32_t THREADS = 32;
const uint32_t BLOCKS = (NUM_BOIDS + (THREADS - 1)) / THREADS;

__global__ void Euclidean_Dist(BoidGPU* a)
{
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	//while (tid < NUM_BOIDS)
	{
		glm::distance(a[x].m_Position, a[y].m_Position);
		//tid += blockDim.x * gridDim.x;
	}
}

int main(void)
{
	OGLRenderer* renderer = OGLRenderer::Instance();
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle();
	BoidGeneratorGPU* boidGPU = new BoidGeneratorGPU(NUM_BOIDS);
	Timer gt;
	cudaSetDevice(0);
	//float* result = new float[RESULT_SIZE];
	BoidGPU* dev_a;
	//float* dev_result;
	
	cudaMalloc((void**)&dev_a, NUM_BOIDS * sizeof(BoidGPU));
	//cudaMalloc((void**)&dev_result, RESULT_SIZE * sizeof(float));

	cudaMemcpy(dev_a, boidGPU->GetBoidData().data(), NUM_BOIDS * sizeof(BoidGPU), cudaMemcpyHostToDevice);

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);
	//Euclidean_Dist << <blocks, threads >> >(dev_a);

	//cudaMemcpy(result, dev_result, RESULT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	//for (unsigned int i = 0; i < NUM_BOIDS * NUM_BOIDS; ++i)
	//{
		//std::cout << result[1] << std::endl;
	//}

	//Main loop.
	while (renderer->ShouldClose()) //Check if the ESC key was pressed or the window was closed
	{
		Euclidean_Dist << <blocks, threads >> >(dev_a);
		gt.startTimer();
		renderer->Render(gt.getLast());
		glfwPollEvents();
		gt.stopTimer();
	}

	OGLRenderer::Release();
	delete boidGPU;
	delete triMesh;
	delete simpleShader;
	cudaDeviceReset();
	cudaFree(dev_a);
	//cudaFree(dev_result);
	return 0;
}
#endif