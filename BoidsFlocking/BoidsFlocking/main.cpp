#include "Common.h"
#include "OGLRenderer.h"
#include "BoidScene.h"
#include "Common.h"
#include "Shader.h"
#include "Mesh.h"

int main(void)
{
	OGLRenderer* renderer = OGLRenderer::Instance();
#if !CUDA
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle(false);
#else
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader_multiDraw.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle(true);
#endif
	int numBoids, algorithm;
	std::cout << "Enter the number of Boids to simulate..." << std::endl;
	std::cin >> numBoids;
	BoidScene* boidScene = new BoidScene(numBoids, simpleShader, triMesh);
	Timer* gt = new Timer;

	renderer->SetCurrentScene(boidScene);

	float frameCount = 0.0f;
	//Main loop.
	while (renderer->ShouldClose()) //Check if the ESC key was pressed or the window was closed
	{
		frameCount += 1.0f;
		gt->startTimer();		
		renderer->Render(gt);
		glfwPollEvents();
		gt->stopTimer();
	} 
	
#if CUDA
	float avgCudaComputeTime = boidScene->GetCUDAElapsedTime() / frameCount;
	std::cout << "CUDA Kernel Average Compute Time: " << avgCudaComputeTime << "ms" << std::endl;
#endif
	std::cout << "CPU Average Compute Time: " << renderer->GetElapsed() / frameCount << "ms" << std::endl;

	delete gt;
	delete triMesh;
	delete simpleShader;	
	delete boidScene;
	OGLRenderer::Release();
	system("pause");
	return 0;
}