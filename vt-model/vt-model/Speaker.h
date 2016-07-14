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
#include <string>
#include <random>
#include <fstream>


class Speaker 
{
public:
    
    // ***** FIXED PARAMETERS FOR MERMELSTEIN'S MODEL ***** //
    double relativeSize;  // Relative size of the parameters. Different for female, male, child.
    // In the larynx.
    Speaker_CordDimensions cord;
    Speaker_CordSpring lowerCord;
    Speaker_CordSpring upperCord;
    Speaker_GlottalShunt shunt;
    // Above the larynx.
    Speaker_Velum velum;
    Speaker_Palate palate;
    Speaker_Tip tip;
    double neutralBodyDistance;
    Speaker_Alveoli alveoli;
    Speaker_TeethCavity teethCavity;
    Speaker_LowerTeeth lowerTeeth;
    Speaker_UpperTeeth upperTeeth;
    Speaker_Lip lowerLip;
    Speaker_Lip upperLip;
    // In the nasal cavity.
    Speaker_Nose nose;
    
    // ***** SIMULATION VARIABLES ***** //
    double fsamp;
    double oversamp;
    long numberOfSamples;
    long sample;
    
    Delta delta; // Delta-tube model and vocal articulation
    int M; // number of tubes
    double art[kArt_muscle_MAX]={0}; // Activations of muscles
    
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
    onebygrad,
    totalVolume;
    
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
    void InitSim(double totalTime, std::string filepath = std::string(),double log_freq = 0);
    void IterateSim();
    bool NotDone() {return (sample < numberOfSamples);}
    double NowSeconds(){return (sample)/fsamp;}
    long Now() {return sample;}
    void setMuscle(int muscle, double position) {art[muscle] = position;}// muscle 0-28, position 0-1
    double getMuscle(int muscle) const {return art[muscle];}
    int Speak();
    int SaveSound(std::string filepath);
    
private:
    void InitializeTube(); // map speaker parameters into delta tube
    void UpdateTube();
    //void UpdateSegment(int m);
    double ComputeSound();
    int InitDataLogger(std::string filepath,double log_freq);
    void Log();
};

/* End of file Speaker.h */
#endif
