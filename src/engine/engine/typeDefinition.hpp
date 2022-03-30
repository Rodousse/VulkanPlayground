#pragma once
#include <memory>

#define PRISMO_DEFINE_HANDLE_TO_RESOURCE(ResourceType, Handle) typedef ResourceType* Handle;

#define PRISMO_NULL_HANDLE nullptr
