//
//  main.cpp
//  vt-model
//
//  Created by William J Wagner on 3/6/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

// #define NDEBUG 1 //disable assertions in the code
// TODO: Create include file and add assert.h. needs to be in only one spot.

#include <iostream>
#include <string>
#include "Speaker.h"
#include "Artword.h"
#include "Control.h"
#include "ArtwordControl.h"
#include "BasePrimControl.h"
#include "functions.h"
#include <yarp/os/all.h>
#include <sys/stat.h>

using namespace std;

void prim_control(Speaker* speaker,double utterance_length, double log_period, std::string params, std::string target) {
    Artword artw = apa();
    Articulation art = {};
    artw.intoArt(art, 0.0);
    BasePrimControl prim(utterance_length,log_period,art,params);
    // Initialize the data logger
    speaker->ConfigDataLogger(target + "primlog" + to_string(1)+ ".log",log_period);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(target + "sound" + to_string(1) + ".log");
}

int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
	rf.configure("ICUB_ROOT",argc,argv);
    rf.setDefaultConfigFile("/home/jacob/Projects/learn-to-speak/bin/conf/prim.ini");

    double sample_freq = rf.find("sample_freq").asDouble();
    int oversamp = rf.find("oversamp").asInt();
    int number_of_glottal_masses = rf.find("glottal_mass").asInt();
    string gender = rf.find("gender").asString().c_str();
    Speaker female(gender,number_of_glottal_masses, sample_freq, oversamp);

    // setup target directory for log files
    string _prefix = rf.find("prefix").asString().c_str();
    string params = rf.find("params").asString().c_str();
    string prefix;

    // create the logs folder if necessary
    mkdir(_prefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);


    int dir_err = -1;
    int index = 0;
    do { 
        index++;
        prefix.clear();
        prefix.append(_prefix);
        prefix.append("prim_gesture");
        prefix.append(to_string(index));
        dir_err = mkdir(prefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    while(dir_err == -1);

    prefix.clear();
    prefix.append(_prefix);
    prefix.append("prim_gesture");
    prefix.append(to_string(index));
    prefix.append("/");

    double utterance_length = rf.find("utterance").asDouble();
    double desired_log_freq = rf.find("log_freq").asDouble();
    int log_period = floor(sample_freq/desired_log_freq);
    double log_freq = sample_freq/log_period;

    // 1.) Create Artword to track
    // 2.) Generate Randomly Stimulated data trials
    // 3.) Perform MATLAB DFA to find primitives and generate Aref of 1.)
    
    // 4.) Perform Primitive Control based on IC only
    prim_control(&female, utterance_length, log_period, params, prefix);
    
    // 5.) Perform Area Function Tracking of 1.)
    //AreaRefControl(&female, log_freq, log_period,prefix);
    
    return 0;
}
