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


ADD_EXECUTABLE(vtSim YarpWrapper.cpp)
ADD_EXECUTABLE(vtSim2 YarpWrapper2.cpp)
ADD_EXECUTABLE(vtDriver YarpDriver.cpp) # don't need all these cpp files linked in

# we now add the YARP and iCub libraries to our project.
TARGET_LINK_LIBRARIES(vtSim vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
TARGET_LINK_LIBRARIES(vtSim2 vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
TARGET_LINK_LIBRARIES(vtDriver vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)

INSTALL(TARGETS vtSim vtSim2 vtDriver DESTINATION /home/jacob/Projects/learn-to-speak/bin)
#INSTALL(TARGETS vtSim DESTINATION ../bin)

