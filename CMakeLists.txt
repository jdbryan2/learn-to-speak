

cmake_minimum_required(VERSION 2.6) #this line is required, I just put some version < current version...

if(COMMAND cmake_policy)
cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

MAKE_DIRECTORY(bin)

PROJECT(VOCAL_TRACT)

#FIND_PACKAGE(YARP)
#FIND_PACKAGE(ICUB)

ADD_SUBDIRECTORY(src)



