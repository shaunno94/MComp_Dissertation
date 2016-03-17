//Author: Shaun Heald
#include "Camera.h"
#include <GLFW/glfw3.h>
#include "Common.h"

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
	look = glm::vec3(0, 0, 1);
	right = glm::vec3(1, 0, 0);
	up = glm::vec3(0, 1, 0);
}

void Camera::UpdateCamera(float dt)
{
	pollKeyBoard(dt);
	pollMouse(dt);

	View = glm::mat4_cast(glm::quat(glm::vec3(glm::radians(pitch), glm::radians(yaw), 0.0))) * glm::translate(glm::mat4(1.0f), position);
	look = glm::normalize(glm::mat3(View) * glm::vec3(0.0, 0.0, 1.0));
	up = glm::normalize(glm::mat3(View) * glm::vec3(0.0, 1.0, 0.0));
	right = glm::normalize(glm::cross(look, up));
	target = position + look;
	View = glm::lookAt(position, target, up);

	VP = Projection * View;
}

//Poll keyboard input.
void Camera::pollKeyBoard(float dt)
{
	float deltaMove = dt * speed;
	// Move forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		position += (look * deltaMove);
	}
	// Move backward
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		position -= (look * deltaMove);
	}
	// Strafe right
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		position += (right * deltaMove);
	}
	// Strafe left
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		position -= (right * deltaMove);
	}
	//move up
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		position += (up * deltaMove);
	}
	//move down
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		position -= (up * deltaMove);
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

	if (yaw < 0.0)
	{
		yaw += 360.0;
	}
	if (yaw > 360.0)
	{
		yaw -= 360.0;
	}
	pitch = min(pitch, 90.0f);
	pitch = max(pitch, -90.0f);
}

Camera::~Camera(){}