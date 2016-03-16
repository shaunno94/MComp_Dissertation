//Author: Shaun Heald
//This class polls input from the keyboard & mouse, it also sets the model view projection matrix.
#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm\gtc\quaternion.hpp>

struct GLFWwindow;

class Camera
{
public:
	Camera(GLFWwindow* window, unsigned int window_height, unsigned int window_width, float FOV, float nearPlane, float farPlane);
	~Camera();
	//Setup MVP matrix
	void UpdateCamera(float dt);
	inline const glm::mat4& GetVP() const { return VP; }

private:
	void pollMouse(float dt);
	void pollKeyBoard(float dt);

	//camera transform variables
	double pitch = 0.0;
	double yaw = 0.0;
	const float speed = 0.1f;
	const float mouseSpeed = 0.3f;

	glm::vec3 position;
	glm::vec3 target;
	glm::quat rotation;
	//Projection, model and view matrices
	glm::mat4 Projection;
	glm::mat4 View;
	glm::mat4 VP;

	//Enviroment variables
	GLFWwindow* window;
	unsigned int window_height;
	unsigned int window_width;
	float half_height;
	float half_width;
};