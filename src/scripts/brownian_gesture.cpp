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
#include "BrownianStim.h"
#include "functions.h"
#include <yarp/os/all.h>
#include <sys/stat.h>

using namespace std;


void brownian_stim_trials(Speaker* speaker, 
                            double utterance_length, 
                            double log_period, 
                            int trials,
                            double delta,
                            double variance,
                            std::string prefix) 
{
    
    BrownianStim bs(utterance_length, delta, variance);

    double chunk_size = utterance_length; //20; // seconds
    double num_chunks = std::ceil(utterance_length/chunk_size);

    // Generate a new random artword
    bs.NewArtword();

    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    Articulation art;
    bs.InitialArt(art);

    // initialize the simulation and tell it how many seconds to buffer
    speaker->InitSim(chunk_size, art);

    for (int trial=1; trial <= num_chunks; trial++)
    {
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "datalog" + to_string(trial)+ ".log",log_period);
        cout << "Trial " << trial << " of " << num_chunks << "\n";
         
        speaker->InitDataLogger();
        
        cout << "Simulating...\n";
        
        while (speaker->LoopBack())
        {
            bs.doControl(speaker);
            // generate the next acoustic sample
            speaker->IterateSim();
        }
        speaker->Speak();
        speaker->SaveSound(prefix + "sound" + to_string(trial) + ".log");
        
    }
    return;
}


int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
	rf.configure("ICUB_ROOT",argc,argv);
    rf.setDefaultConfigFile("/home/jacob/Projects/learn-to-speak/bin/conf/brownian.ini");

    double sample_freq = rf.find("sample_freq").asDouble();
    int oversamp = rf.find("oversamp").asInt();
    int number_of_glottal_masses = rf.find("glottal_mass").asInt();
    string gender = rf.find("gender").asString().c_str();
    Speaker female(gender,number_of_glottal_masses, sample_freq, oversamp);

    // setup target directory for log files
    string _prefix = rf.find("target").asString().c_str();
    string prefix;

    int dir_err = -1;
    int index = 0;
    do { 
        index++;
        prefix.clear();
        prefix.append(_prefix);
        prefix.append("brownian_gesture");
        prefix.append(to_string(index));
        dir_err = mkdir(prefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    while(dir_err == -1);

    prefix.clear();
    prefix.append(_prefix);
    prefix.append("brownian_gesture");
    prefix.append(to_string(index));
    prefix.append("/");

    double utterance_length = rf.find("utterance").asDouble();
    double desired_log_freq = rf.find("log_freq").asDouble();
    int log_period = floor(sample_freq/desired_log_freq);
    double log_freq = sample_freq/log_period;

    int trials = rf.find("trials").asInt();

    double delta = rf.find("delta").asDouble();
    double variance = rf.find("variance").asDouble();
    
    
    // 2.) Generate Randomly Stimulated data trials
    brownian_stim_trials(&female,utterance_length,log_period,trials, delta, variance, prefix);
    
    return 0;
}
