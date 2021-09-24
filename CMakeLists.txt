cmake_minimum_required(VERSION 2.8.9)
project(components)

find_package(MPI REQUIRED)

set(OMPI_CC gcc-8)
set(OMPI_CXX g++-8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif (NOT CMAKE_BUILD_TYPE)

set(KACOMP_DEFINITIONS "")
option(KACOMP_TIEBREAK_DEGREE "Tiebreak based on degree instead of ID." OFF)
option(KACOMP_MEMINFO "Show memory consumed." OFF)

if(KACOMP_TIEBREAK_DEGREE)
  set(KACOMP_DEFINITIONS "${KACOMP_DEFINITIONS} -DTIEBREAK_DEGREE")
endif()
if(KACOMP_MEMINFO)
  set(KACOMP_DEFINITIONS "${KACOMP_DEFINITIONS} -DMEMINFO")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KACOMP_DEFINITIONS}")

include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -std=c++14 -fno-omit-frame-pointer -march=native -DNDEBUG -DNSTATUS -DNWARMUP -Wall -fPIC -O3 -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -fsanitize=address -std=c++14 -fno-omit-frame-pointer -march=native -fno-stack-limit -Wall -fPIC -Og -g")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")

add_subdirectory(extern/kagen)
# add_subdirectory(extern/KaHIP)

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
          lib/communication/vertex_comm.cpp
          lib/components
          lib/contraction
          lib/data_structures
          lib/data_structures/dynamic_graph_comm.cpp
          lib/data_structures/semidynamic_graph_comm.cpp
          lib/data_structures/static_graph_comm.cpp
          lib/io
          lib/tools
)

add_executable(exponential ${SOURCES} "app/exp_contraction.cpp")
# add_executable(allreduce ${SOURCES} "app/all_reduce.cpp")
# add_executable(local ${SOURCES} "app/local_contraction.cpp")
# add_executable(shortcuts ${SOURCES} "app/shortcuts.cpp")
# add_executable(labels ${SOURCES} "app/labels.cpp")

set(LIBS
        spooky
        sampling
        ${MPI_LIBRARIES}
        # asan
        )

target_link_libraries(exponential LINK_PUBLIC ${LIBS})
# target_link_libraries(allreduce LINK_PUBLIC ${LIBS})
# target_link_libraries(local LINK_PUBLIC ${LIBS})
# target_link_libraries(shortcuts LINK_PUBLIC ${LIBS})
# target_link_libraries(labels LINK_PUBLIC ${LIBS})
