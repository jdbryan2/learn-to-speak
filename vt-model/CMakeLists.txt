
cmake_minimum_required(VERSION 2.6) #this line is required, I just put some version < current version...

if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# add required linker flags

add_definitions(-std=c++11)
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ICUB_LINK_FLAGS} -lrt -lm -lasound -pthread")              
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -funroll-loops")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -funroll-loops")


SET(VTSIM_FILES 
    Articulation_enums.h
    ArtwordControl.cpp ArtwordControl.h
    Artword.cpp Artword.h
    Artword_def.h
    BasePrimControl.cpp BasePrimControl.h
    BrownianStim.cpp BrownianStim.h
    Control.h
    Delta.h
    RandomStim.cpp RandomStim.h
    Sound.cpp Sound.h
    Speaker.cpp Speaker.h
    Speaker_def.h
    VocalTract.cpp VocalTract.h)

ADD_LIBRARY(vtModel ${VTSIM_FILES})

# we now add the YARP and iCub libraries to our project.
TARGET_LINK_LIBRARIES(vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)

ADD_EXECUTABLE(vocal_tract main.cpp)
TARGET_LINK_LIBRARIES(vocal_tract vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio)
INSTALL(TARGETS vocal_tract DESTINATION /home/jacob/Projects/learn-to-speak/bin)
