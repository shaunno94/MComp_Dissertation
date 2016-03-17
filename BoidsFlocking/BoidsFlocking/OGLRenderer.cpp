#include "OGLRenderer.h"
#include "Shader.h"
#include "Scene.h"
#include <glm/gtc/type_ptr.hpp>

OGLRenderer* OGLRenderer::instance = nullptr;

OGLRenderer* OGLRenderer::Instance()
{
	if (!instance)
	{
		instance = new OGLRenderer();
	}
	return instance;
}

void OGLRenderer::Release()
{
	if (instance)
	{
		delete instance;
		instance = nullptr;
	}
}

OGLRenderer::OGLRenderer()
{	
	init_glfw();
	init_glew();
	currentScene = nullptr;
	currentShader = nullptr;
}

OGLRenderer::~OGLRenderer()
{
	if (window)
	{
		glfwDestroyWindow(window);
	}
}

void OGLRenderer::init_glew()
{
	glewExperimental = GL_TRUE;

	// Initialise GLEW
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Error: Failed to initialise GLEW." << std::endl;
		system("pause");
		exit(1);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_CLAMP);
	glDepthFunc(GL_LEQUAL);
	glClearDepth(1.0f);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void OGLRenderer::init_glfw()
{
	// Initialise GLFW
	if (!glfwInit())
	{
		std::cerr << "Error: Failed to initialise GLFW." << std::endl;
		system("pause");
		exit(1);
	}

	//Set window variables
	MODE = glfwGetVideoMode(glfwGetPrimaryMonitor());

	glfwWindowHint(GLFW_SAMPLES, 8); // 8x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); //OpenGL 4.2
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_DEPTH_BITS, 32);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);
	
	if (window == nullptr)
	{
		std::cerr << "Error: Failed to open GLFW window. OpenGL 4.2 Required." << std::endl;
		system("pause");
		glfwTerminate();
		exit(1);
	}

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS | GLFW_CURSOR_DISABLED, GL_TRUE);
}

void OGLRenderer::SetCurrentShader(Shader* s)
{
	currentShader = s;
	glUseProgram(currentShader->GetShaderProgram());
	UpdateShaderMatrices();
}

void OGLRenderer::Render(float dt)
{
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	if (currentScene)
	{
		currentScene->UpdateScene(dt);
		currentScene->RenderScene();
	}

	glfwSwapBuffers(window);
}

bool OGLRenderer::ShouldClose()
{
	return (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) && (glfwWindowShouldClose(window) == 0);
}

void OGLRenderer::UpdateShaderMatrices()
{
	glUniformMatrix4fv(currentShader->GetVPMatrixLoc(), 1, GL_FALSE, glm::value_ptr(currentScene->GetCamera()->GetVP()));
}