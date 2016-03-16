#pragma once

class Entity;
class Mesh;
class Shader;

class RenderComponent
{
public:
	RenderComponent(Mesh* mesh, Shader* shader);
	~RenderComponent();

	void Draw();
	void SetParent(Entity* e);

private:
	Entity* m_Entity;
	Mesh* m_Mesh;
	Shader* m_Shader;
};