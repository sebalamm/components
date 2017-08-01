cmake_minimum_required(VERSION 2.8.9)
project(components)

set(OMPI_CXX g++)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -fno-omit-frame-pointer -Wall -g -O3 -std=c++11")

include_directories(
    app
    lib
    lib/communication
    lib/components
    lib/data_structures
    lib/io
    lib/tools
)

file(GLOB COMM "lib/communication/*.cpp")
file(GLOB DS "lib/data_structures/*.cpp")
file(GLOB COMP "lib/components/*.cpp")
file(GLOB IO "lib/io/*.cpp")

set(SOURCES 
  ${COMM}
  ${DS}
  ${COMP}
  ${IO}
)

add_executable(components ${SOURCES} "app/components.cpp")

set(LIBS 
    gmp
)
 
target_link_libraries(components LINK_PUBLIC ${LIBS})
