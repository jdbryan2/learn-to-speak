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
#include <random>
#include <fstream>


class Speaker : private VocalTract, private Delta {
public:
    
    Articulation art ={0}; // Activations of muscles
    
    // ***** SIMULATION VARIABLES ***** //
    double fsamp;
    double oversamp;
    long numberOfSamples;
    long sample;
    
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
    double logfreq;
    long numberOfLogSamples;
    int numberOfOversampLogSamples;
    int logCounter;
    long logSample;
    Sound *result;

    Speaker(std::string kindOfSpeaker, int numberOfVocalCordMasses, double samplefreq, int oversamplefreq);
    ~Speaker() { delete result;}
    void InitSim(double totalTime, Articulation initialArt);
    int InitDataLogger(std::string filepath,double log_freq);
    void IterateSim();
    bool NotDone() {return (sample < numberOfSamples);}
    double NowSeconds(){return (sample)/fsamp;}
    long Now() {return sample;}
    void setMuscle(int muscle, double position) {art[muscle] = position;}// muscle 0-28, position 0-1
    double getMuscle(int muscle) const {return art[muscle];}
    int Speak();
    int SaveSound(std::string filepath);
    double getVolume();
    
private:
    void InitializeTube(); // map speaker parameters into delta tube
    void UpdateTube();
    double ComputeSound();
    void Log();
};

/* End of file Speaker.h */
#endif
