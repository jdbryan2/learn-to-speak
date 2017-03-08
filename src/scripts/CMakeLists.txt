cmake_minimum_required(VERSION 2.6) #this line is required, I just put some version < current version...

if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


PROJECT(VOCAL_TRACT)

FIND_PACKAGE(YARP)
FIND_PACKAGE(ICUB)

# add include directories
INCLUDE_DIRECTORIES(${YARP_INCLUDE_DIRS} ${ICUB_INCLUDE_DIRS} ../vt-model)

add_definitions(-std=c++11)

# add required linker flags
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ICUB_LINK_FLAGS} -lrt -lm -lasound -pthread")              
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -funroll-loops")

ADD_EXECUTABLE(main main.cpp)
ADD_EXECUTABLE(rand_gesture rand_gesture.cpp)
ADD_EXECUTABLE(brownian_gesture brownian_gesture.cpp)
ADD_EXECUTABLE(prim_control prim_control.cpp)

# we now add the YARP and iCub libraries to our project.
TARGET_LINK_LIBRARIES(main vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
TARGET_LINK_LIBRARIES(rand_gesture vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
TARGET_LINK_LIBRARIES(brownian_gesture vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
TARGET_LINK_LIBRARIES(prim_control vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)

INSTALL(TARGETS main rand_gesture brownian_gesture prim_control DESTINATION /home/jacob/Projects/learn-to-speak/bin)
#INSTALL(TARGETS vtSim DESTINATION ../bin)

