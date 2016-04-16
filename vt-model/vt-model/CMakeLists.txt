

cmake_minimum_required(VERSION 2.0) #this line is required, I just put some version < current version...

if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT(VOCAL_TRACT)

FIND_PACKAGE(YARP)
FIND_PACKAGE(ICUB)


# add include directories
INCLUDE_DIRECTORIES(${YARP_INCLUDE_DIRS} ${ICUB_INCLUDE_DIRS} )
#INCLUDE_DIRECTORIES(${YARP_INCLUDE_DIRS} ${ICUB_INCLUDE_DIRS} /home/jacob/Projects/learn-to-speak/vt-model/external_libs/portaudio)

add_definitions(-std=c++11)
# add required linker flags
#SET(COMPILE_FLAGS "-std=c++11")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ICUB_LINK_FLAGS} -lrt -lm -lasound -pthread")              
#ADD_LIBRARY(portaudio /home/jacob/Projects/learn-to-speak/vt-model/external_libs/portaudio/portaudio.h)
#ADD_LIBRARY(portaudio STATIC IMPORTED)
#SET_TARGET_PROPERTIES(portaudio PROPERTIES IMPORTED_LOCATION /home/jacob/Projects/learn-to-speak/vt-model/external_libs/portaudio/libportaudio.a)

ADD_EXECUTABLE(vocal_tract main.cpp Speaker.cpp Delta.cpp Artword.cpp Sound.cpp support_functions.cpp )

# we now add the YARP and iCub libraries to our project.
TARGET_LINK_LIBRARIES(vocal_tract ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
#TARGET_LINK_LIBRARIES(vocal_tract /home/jacob/Projects/learn-to-speak/vt-model/external_libs/portaudio/libportaudio.a)
