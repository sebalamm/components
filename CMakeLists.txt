cmake_minimum_required(VERSION 2.8.9)
project(components)

find_package(MPI REQUIRED)

set(OMPI_CC gcc-7)
set(OMPI_CXX g++-7)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Wall -g3 -std=c++11")
# set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsanitize=address ${MPI_LINK_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -Wall -g3 -std=c++11")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")

include_directories(
        app
        lib
        lib/communication
        lib/components
        lib/contraction
        lib/data_structures
        lib/io
        lib/tools
)


set(SOURCES
        app
        lib
        lib/communication
        lib/communication/node_communicator.cpp
        lib/components
        lib/contraction
        lib/data_structures
        lib/data_structures/graph_access.cpp
        lib/io
        lib/tools
        )

add_executable(components ${SOURCES} "app/components.cpp")

set(LIBS
        gmp
        ${MPI_LIBRARIES}
        )

target_link_libraries(components LINK_PUBLIC ${LIBS})
