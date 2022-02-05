
set(APP_SRC 
      main.cpp)

add_executable(app ${APP_SRC})
target_link_libraries(app PRIVATE imgui engine) 

