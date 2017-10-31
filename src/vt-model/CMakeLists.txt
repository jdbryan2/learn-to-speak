
cmake_minimum_required(VERSION 2.6) #this line is required, I just put some version < current version...

if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# add required linker flags
find_package(Boost COMPONENTS filesystem system date_time python3 numpy REQUIRED)
message("Include dirs of boost: " ${Boost_INCLUDE_DIRS} )
message("Libs of boost: " ${Boost_LIBRARIES} )

find_package(PythonLibs REQUIRED)
message("Include dirs of Python: " ${PYTHON_INCLUDE_DIRS} )
message("Libs of Python: " ${PYTHON_LIBRARIES} )

find_package(gsl REQUIRED)
message("Include dirs of GSL: " ${GSL_INCLUDE_DIRS} )
message("Libs of GSL: " ${GSL_LIBRARIES} )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
)


add_definitions(-std=c++11)
add_definitions(-DBOOST_ALL_NO_LIB )
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ICUB_LINK_FLAGS} -lrt -lm -lasound -pthread ")              
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -DNORT -lm -O3 -funroll-loops")              
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -funroll-loops")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -O3 -funroll-loops")

SET(VTSIM_FILES 
    Articulation_enums.h
    ArtwordControl.cpp ArtwordControl.h
    Artword.cpp Artword.h
    Artword_def.h
    BasePrimControl_jdb.cpp BasePrimControl.h
    BrownianStim.cpp BrownianStim.h
    Control.h
    Delta.h
    RandomStim.cpp RandomStim.h
    Sound.cpp Sound.h
    Speaker.cpp Speaker.h
    Speaker_def.h
    VocalTract.cpp VocalTract.h)

ADD_LIBRARY(vtModel ${VTSIM_FILES})

ADD_LIBRARY(PyRAAT MODULE Speaker.cpp VocalTract.cpp Sound.cpp)
ADD_LIBRARY(Artword MODULE Artword.cpp)

# we now add the YARP and iCub libraries to our project.
#TARGET_LINK_LIBRARIES(vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#TARGET_LINK_LIBRARIES(PyRAAT ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#TARGET_LINK_LIBRARIES(Artword ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(vtModel portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(PyRAAT portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(Artword portaudio ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(PyRAAT PROPERTIES PREFIX "" )
set_target_properties(Artword PROPERTIES PREFIX "" )

ADD_EXECUTABLE(vocal_tract main.cpp)
#TARGET_LINK_LIBRARIES(vocal_tract vtModel ${YARP_LIBRARIES} ${ICUB_LIBRARIES} portaudio )
TARGET_LINK_LIBRARIES(vocal_tract vtModel portaudio ${GSL_LIBRARIES})
#INSTALL(TARGETS vocal_tract DESTINATION /home/jacob/Projects/learn-to-speak/bin)
#INSTALL(TARGETS PyRAAT DESTINATION /home/jacob/Projects/learn-to-speak/bin)
#INSTALL(TARGETS Artword DESTINATION /home/jacob/Projects/learn-to-speak/bin)
INSTALL(TARGETS vocal_tract DESTINATION /Users/JacobWagner/Documents/Repositories/learn-to-speak/bin/)
INSTALL(TARGETS PyRAAT DESTINATION /Users/JacobWagner/Documents/Repositories/learn-to-speak/bin/)
INSTALL(TARGETS Artword DESTINATION /Users/JacobWagner/Documents/Repositories/learn-to-speak/bin/)
