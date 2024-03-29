#include "RenderComponent.h"
#include "Shader.h"
#include "Entity.h"
#include "Mesh.h"
#include "OGLRenderer.h"
#include <glm/gtc/type_ptr.hpp>

RenderComponent::RenderComponent(Mesh* mesh, Shader* shader)
{
	m_Mesh = mesh;
	m_Shader = shader;
}

RenderComponent::~RenderComponent()
{

}

void RenderComponent::SetParent(Entity* e)
{
	m_Entity = e;
}

void RenderComponent::Draw()
{
	OGLRenderer::Instance()->SetCurrentShader(m_Shader);
#if !CUDA
	glUniformMatrix4fv(m_Shader->GetModelMatrixLoc(), 1, GL_FALSE, (float*)m_Entity->GetWorldTransform());
#endif
	m_Mesh->Draw();
}