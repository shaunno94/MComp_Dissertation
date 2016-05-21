#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "Timer.h"

class Scene;
class Shader;

class OGLRenderer
{
public:
	static OGLRenderer* Instance();
	static void Release();

	void Render(Timer* t);
	bool ShouldClose();
	void SetCurrentShader(Shader* s);
	void SetCurrentScene(Scene* s) { currentScene = s; }
	GLFWwindow* GetWindow() const { return window; }
	unsigned int GetWindowWidth() const { return WIDTH; }
	unsigned int GetWindowHeight() const { return HEIGHT; }
	unsigned int GetSSBO_ID() const { return SSBO; }
	float GetElapsed() const { return elapsedCPU; }

private:
	OGLRenderer();
	~OGLRenderer();
	void init_glew();
	void init_glfw();

	const GLFWvidmode* MODE;
	//Pointer to current GLFW window object.
	GLFWwindow* window;

	Shader* currentShader;
	Scene* currentScene;

	static OGLRenderer* instance;
	float elapsedCPU = 0.0f;

	unsigned int SSBO;

	const unsigned int WIDTH = 1600;
	const unsigned int HEIGHT = 900;
	const char* TITLE = "MComp Dissertation - Boids Simulation";
};