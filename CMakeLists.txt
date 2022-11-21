cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(predator_prey)


# Include what you use detector
find_program(IWYU NAMES include-what-you-use)
if(IWYU)
  message(STATUS "executing include-what-you-use")
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU})
endif()

# Set parameters
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the executable
file(GLOB SOURCES src/*.cpp include/*.h)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE src/ include/)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_compile_options(
  ${PROJECT_NAME}
  PUBLIC -Werror
         -Wall
         -Wextra
         -O3
         -flto=auto 
         -fpic
         -pthread
         -ffunction-sections
         -fdata-sections)

# Link
add_subdirectory(lib/robosimcpp)
target_link_libraries(${PROJECT_NAME} PRIVATE lib/robosim)
target_include_directories(${PROJECT_NAME} PRIVATE lib/robosimcpp/include)

# torch
find_package(Torch REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# Link

# The following code block is suggested to be used on Windows. According to
# https://github.com/pytorch/pytorch/issues/25457, the DLLs need to be copied to
# avoid memory errors.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll
    ")
  add_custom_command(
    TARGET example-app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif(MSVC)
