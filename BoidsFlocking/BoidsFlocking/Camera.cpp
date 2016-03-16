//Author: Shaun Heald
#include "Camera.h"
#include <GLFW/glfw3.h>

Camera::Camera(GLFWwindow* window, unsigned int window_height, unsigned int window_width, float FOV, float nearPlane, float farPlane)
{
	this->window = window;
	this->window_height = window_height;
	this->window_width = window_width;
	half_width = window_width / 2.0f;
	half_height = window_height / 2.0f;
	Projection = glm::perspective(FOV, GLfloat(window_width) / GLfloat(window_height), nearPlane, farPlane);
	position = glm::vec3(0.0, 0.0, 0.0);
	target = glm::vec3(0.0, 0.0, 0.0);
	rotation = glm::quat(0, 0, 0, 1.0);
}

void Camera::UpdateCamera(float dt)
{
	pollKeyBoard(dt);
	pollMouse(dt);

	rotation = glm::quat(pitch, yaw, 0, 1.0);
	target = glm::vec3(glm::vec4(1, 1, 1, 1) * (glm::mat4_cast(rotation) * glm::translate(glm::mat4(1.0f), position)));
	//Setup view matrix. 
//	View = glm::lookAt(position, target, glm::vec3(0, 1, 0));
	//View = glm::rotate(float(-pitch), glm::vec3(1, 0, 0)) * glm::rotate(float(-yaw), glm::vec3(0, 1, 0)) * glm::translate(-position);
	//VP = Projection * View;
	VP = glm::mat4(1.0);
}

//Poll keyboard input.
void Camera::pollKeyBoard(float dt)
{
	float deltaMove = dt * speed;
	// Move forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position += (glm::vec3(0.0f, 0.0f, 1.0f) * deltaMove);
	}
	// Move backward
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position -= (glm::vec3(0.0f, 0.0f, 1.0f) * deltaMove);
	}
	// Strafe right
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position -= (glm::vec3(1.0f, 0.0f, 0.0f) * deltaMove);
	}
	// Strafe left
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position += (glm::vec3(1.0f, 0.0f, 0.0f) * deltaMove);
	}
	//move up
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		position.y += deltaMove;
	}
	//move down
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		position.y -= deltaMove;
	}
}

//Poll mouse input.
void Camera::pollMouse(float dt)
{
	double x, y;
	float deltaMove = dt * mouseSpeed;
	glfwGetCursorPos(window, &x, &y);
	pitch += deltaMove * (half_height - y);
	yaw += deltaMove * (half_width - x);
	glfwSetCursorPos(window, half_width, half_height);
}

Camera::~Camera(){}