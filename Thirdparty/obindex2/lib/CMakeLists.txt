cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(obindex2 LANGUAGES CXX)

# Compile tests or not
option(EnableTesting "EnableTesting" OFF)

# Setting the flags for profiling information or not
if(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Setting Release options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Setting Debug options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1 -pg -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -pg -march=native")
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include GNUInstallDirs to set the installation directories
include(GNUInstallDirs)

# Packages required to work with the library
find_package(OpenCV REQUIRED) # OpenCV
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(OpenMP REQUIRED) # OpenMP
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

### Targets ###
add_library(obindex2
        SHARED
            src/binary_descriptor.cc
            src/binary_tree_node.cc
            src/binary_tree.cc
            src/binary_index.cc
)
target_link_libraries(obindex2
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
)
target_include_directories(obindex2
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS obindex2
        EXPORT obindex2Config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/obindex2
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

export(TARGETS
       obindex2
       NAMESPACE obindex2::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/obindex2Config.cmake"
)

install(EXPORT
        obindex2Config
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/obindex2/cmake"
        NAMESPACE obindex2::
)