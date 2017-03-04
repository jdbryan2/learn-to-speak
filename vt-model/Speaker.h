#ifndef _Speaker_h_
#define _Speaker_h_
/* Speaker.h
 *
 * Copyright (C) 1992-2011,2015 Paul Boersma
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "Speaker_def.h"
#include "Delta.h"
#include "Articulation_enums.h"
#include "Sound.h"
#include "VocalTract.h"
#include <string>
#include <iostream>
#include <random>
#include <fstream>
// boost stuff
#include <boost/python/list.hpp>
#include <boost/python/class.hpp>

using namespace std;

typedef double AreaFcn[MAX_NUMBER_OF_TUBES];
typedef double PressureFcn[MAX_NUMBER_OF_TUBES];


class Speaker : private VocalTract, private Delta {
public:
    Articulation art ={0}; // Activations of muscles
    
    // ***** SIMULATION VARIABLES ***** //
    double fsamp;
    double oversamp;
    long numberOfSamples;
    long sample;
    long loop_count;
    double totalTime;
    Sound *result;
    
    double Dt,
    rho0,
    c,
    onebyc2,
    rho0c2,
    halfDt,
    twoDt,
    halfc2Dt,
    twoc2Dt,
    onebytworho0,
    Dtbytworho0;
    
    double tension,
    rrad,
    onebygrad;
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
    
    // ***** DATA LOGGING VARIABLES ***** //
    bool log_data = false;
    std::ofstream * log_stream = nullptr;
    int log_period;
    long numberOfLogSamples;
    int logCounter;
    long logSample;
    
public:
    Speaker(std::string kindOfSpeaker, int numberOfVocalCordMasses, double samplefreq, int oversample_multiplier);
    ~Speaker() {cout << "end speaker 0" << endl;  delete result; cout << "end speaker 1" << endl; delete log_stream; cout << "end speaker 2" << std::endl; }
    void InitSim(double totalTime, Articulation initialArt);
    int ConfigDataLogger(std::string filepath,int _log_period);
    void IterateSim();
    bool NotDone() {return (sample < numberOfSamples);}
    double NowSeconds(){return (sample)/fsamp;}
    double NowSecondsLooped(){return (numberOfSamples+sample)/fsamp;}
    long Now() {return sample;}
    void setMuscle(int muscle, double position) {art[muscle] = position;}// muscle 0-28, position 0-1
    double getMuscle(int muscle) const {return art[muscle];}
    int Speak();
    int SaveSound(std::string filepath);
    double getVolume();
    void getAreaFcn(AreaFcn AreaFcn_);
    void getPressureFcn(PressureFcn PressureFcn_);

    float getLastSample() {return result->z[sample-1];}
    int LoopBack() { if(NotDone()) { return 1;} else {sample = 0; loop_count++;return 0;} }
    void InitDataLogger();

    // Boost wrappers
    void py_InitSim(double totalTime, boost::python::list initialArtList);
    boost::python::list py_getAreaFcn();
    boost::python::list py_getPressureFcn();

    
private:
    void InitializeTube(); // map speaker parameters into delta tube
    void UpdateTube();
    double ComputeSound();
    //void InitDataLogger();
    void Log();
};

// Boost it up!

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(speaker) // tells boost where to look
{
   class_<Speaker>("Speaker", init<std::string, int, double, int>())
       .def("InitSim", &Speaker::py_InitSim)
       .def("IterateSim", &Speaker::IterateSim)
       .def("getAreaFcn", &Speaker::py_getAreaFcn)
       .def("Now", &Speaker::Now)
       .def("getLastSample", &Speaker::getLastSample)
       .def("NotDone", &Speaker::NotDone)
       .def("setMuscle", &Speaker::setMuscle)
       .def("getMuscle", &Speaker::getMuscle)
    ;
}

/* End of file Speaker.h */
#endif
