#version 450
//gl_DrawIDARB
#extension GL_ARB_shader_draw_parameters : enable

layout(std430, binding=6) buffer ModelMatrixSSBO
{
	mat4 mMatrix[];
};

uniform mat4 VPMatrix;

in vec3 position;

void main()
{
	mat4 MVP = VPMatrix * mMatrix[gl_DrawIDARB];
	gl_Position = MVP * vec4(position, 1.0);
}