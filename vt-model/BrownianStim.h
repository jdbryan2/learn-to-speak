//
//  RandomStim.hpp
//  vt-model
//
//  Created by William J Wagner on 9/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef BrownianStim_h
#define BrownianStim_h

#include "ArtwordControl.h"
#include "Artword.h"
#include "Articulation_enums.h"

#define BROWN_ART 29

class BrownianStim : public ArtwordControl {
public:
    BrownianStim(double utterance_length, 
               double delta, // size of timestep in random walk
               double variance);
               
    void NewArtword();
    void doControl(Speaker* speaker);
    void InitialArt(Articulation art);
private:
    void CreateArtword();
private:
    int arts[BROWN_ART];
    double delta;
    double variance;
    // TODO: May want to set the seed for generator.
    std::default_random_engine generator;
    std::normal_distribution<double> walk; // distribution that drives how much the parameters walk with each time step
};

#endif /* RandomStim_h */
