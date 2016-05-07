#include "Common.h"
#include "OGLRenderer.h"
#include "BoidScene.h"
#include "Timer.h"
#include "Common.h"
#include "Shader.h"
#include "Mesh.h"

int main(void)
{
	OGLRenderer* renderer = OGLRenderer::Instance();
	Shader* simpleShader = new Shader(SHADER_DIR"vertex_shader.glsl", SHADER_DIR"frag_shader.glsl");
	Mesh* triMesh = Mesh::GenerateTriangle();
	BoidScene* boidScene = new BoidScene(NUM_BOIDS, simpleShader, triMesh);
	Timer gt;

	renderer->SetCurrentScene(boidScene);

	//Main loop.
	while (renderer->ShouldClose()) //Check if the ESC key was pressed or the window was closed
	{
		gt.startTimer();		
		renderer->Render(gt.getLast());
		glfwPollEvents();
		gt.stopTimer();
	} 
	
	delete triMesh;
	delete simpleShader;	
	delete boidScene;
	OGLRenderer::Release();

	return 0;
}