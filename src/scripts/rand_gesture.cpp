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
#include "RandomStim.h"
#include "functions.h"
#include <yarp/os/all.h>
#include <sys/stat.h>

using namespace std;

void random_stim_trials(Speaker* speaker,double utterance_length, double log_period, int trials, std::string prefix) 
{
    std::normal_distribution<double>::param_type hold_time_param(0.1,0.1);
    std::uniform_real_distribution<double>::param_type activation_param(0.0,1.0);
    RandomStim rs(utterance_length, speaker->fsamp, hold_time_param, activation_param);
    for (int trial=1; trial <= trials; trial++)
    {
        // Generate a new random artword
        rs.NewArtword();
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "datalog" + to_string(trial)+ ".log",log_period);
        cout << "Trial " << trial << "\n";
        simulate(speaker, &rs);
        speaker->Speak();
        speaker->SaveSound(prefix + "sound" + to_string(trial) + ".log");
    }
}


int main(int argc, char *argv[])
{
    yarp::os::ResourceFinder rf;
	rf.configure("ICUB_ROOT",argc,argv);
    rf.setDefaultConfigFile("/home/jacob/Projects/learn-to-speak/bin/conf/gesture.ini");

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
        prefix.append("rand_gesture");
        prefix.append(to_string(index));
        dir_err = mkdir(prefix.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        cout << index <<  dir_err << endl;
    }
    while(dir_err == -1);

    prefix.clear();
    prefix.append(_prefix);
    prefix.append("rand_gesture");
    prefix.append(to_string(index));
    prefix.append("/");

    double utterance_length = rf.find("utterance").asDouble();
    double desired_log_freq = rf.find("log_freq").asDouble();
    int log_period = floor(sample_freq/desired_log_freq);
    double log_freq = sample_freq/log_period;

    int trials = rf.find("trials").asInt();
    
    // 2.) Generate Randomly Stimulated data trials
    random_stim_trials(&female,utterance_length,log_period, trials, prefix);
    //brownian_stim_trials(&female,utterance_length,log_period,prefix);
    
    return 0;
}
