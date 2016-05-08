#include "Entity.h"

unsigned int Entity::id = 0;

Entity::Entity(const std::string& name)
{
	m_Parent = nullptr;
	m_Name = name;
	m_CamDist = 0.0f;
	id++;

	m_LocalTransform = glm::mat4(1.0f);
	m_WorldTransform = glm::mat4(1.0f);
}

Entity::~Entity()
{
	/*if (m_RenderComponent)
	{
		delete m_RenderComponent;
		m_RenderComponent = nullptr;
	}*/
}

Entity*	Entity::FindEntity(const std::string& name)
{
	//Has this object got the same name?
	if (m_Name.compare(name) == 0)
	{
		return this;
	}

	//Recursively search ALL child objects and return the first one matching the given name
	for (auto child : m_Children)
	{
		//Has the object in question got the same name?
		Entity* cObj = child->FindEntity(name);
		if (cObj)
		{
			return cObj;
		}
	}

	//Object not found with the given name
	return nullptr;
}

void Entity::AddChildObject(Entity* child)
{
	m_Children.push_back(child);
	child->m_Parent = this;
}

void Entity::OnRenderObject()
{
	if (m_RenderComponent)
		m_RenderComponent->Draw();
}

void Entity::OnUpdateObject(float dt)
{
	m_WorldTransform = m_LocalTransform;

	if (m_Parent)
		m_WorldTransform = m_Parent->m_WorldTransform * m_WorldTransform;

	m_WorldTransformPtr = &m_WorldTransform;

	for (auto child : m_Children)
	{
		child->OnUpdateObject(dt);
	}
}