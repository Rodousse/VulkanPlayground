#pragma once
#include <iostream>

#define LOG(level, stream, message) stream << "[" << __func__ << "]: [" << level << "]" << message << '\n';

#define LOG_INFO(message) LOG("info", std::cout, message)
#define LOG_WARNING(message) LOG("warning", std::cout, message)
#define LOG_ERROR(message) LOG("error", std::cerr, message)
#define THROW(exception) \
    LOG_ERROR(exception.what()); \
    throw exception;
