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
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Wall -g")
# set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS} -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -Wall -g")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")

add_subdirectory(extern/kagen)

include_directories(
        app
        lib
        lib/communication
        lib/components
        lib/contraction
        lib/data_structures
        lib/io
        lib/tools
        extern/kagen/extlib/sampling/sampling
        extern/kagen/include
        extern/kagen/include/generators
        extern/kagen/include/generators/barabassi
        extern/kagen/include/generators/geometric
        extern/kagen/include/generators/geometric/delaunay
        extern/kagen/include/generators/geometric/rgg
        extern/kagen/include/generators/gnm
        extern/kagen/include/generators/gnp
        extern/kagen/include/generators/hyperbolic
        extern/kagen/include/io
        extern/kagen/include/tools
        extern/kagen/interface/
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
add_executable(shortcuts ${SOURCES} "app/shortcuts.cpp")

set(LIBS
        gmp
        spooky
        sampling
        ${MPI_LIBRARIES}
        )

target_link_libraries(components LINK_PUBLIC ${LIBS})
target_link_libraries(shortcuts LINK_PUBLIC ${LIBS})
