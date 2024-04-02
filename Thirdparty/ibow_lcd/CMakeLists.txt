cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(ibow_lcd LANGUAGES CXX)

message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Setting the flags for profiling information or not
if (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Setting Release options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Setting Debug options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1 -pg -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -pg -march=native")
endif ()

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)

# Include GNUInstallDirs to set the installation directories
include(GNUInstallDirs)

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Other packages
find_package(OpenCV REQUIRED) # OpenCV
find_package(Boost REQUIRED COMPONENTS system filesystem)
find_package(obindex2 REQUIRED) # obindex2
find_package(OpenMP REQUIRED) # OpenMP
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

### Targets ###
# Library
add_library(lcdetector
        SHARED
            src/lcdetector.cc
)
target_link_libraries(lcdetector
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        obindex2::obindex2
)
target_include_directories(lcdetector
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS lcdetector
        EXPORT lcdetectorConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ibow_lcd
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/)

export(TARGETS
       lcdetector
       NAMESPACE lcdetector::
       FILE "${CMAKE_CURRENT_BINARY_DIR}/lcdetectorConfig.cmake"
)

install(EXPORT
        lcdetectorConfig
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/lcdetector/cmake"
        NAMESPACE lcdetector::
)