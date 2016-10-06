//
//  RandomStim.hpp
//  vt-model
//
//  Created by William J Wagner on 9/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef RandomStim_h
#define RandomStim_h

#include "ArtwordControl.h"
#include "Artword.h"
#include "Articulation_enums.h"

#define NUM_ART 29

class RandomStim : public ArtwordControl {
public:
    RandomStim(double utterance_length, double sample_freq,
               const std::normal_distribution<double>::param_type hold_time_param,
               const std::uniform_real_distribution<double>::param_type activation_param);
    void NewArtword();
    void doControl(Speaker* speaker);
    void InitialArt(Articulation art);
private:
    void CreateArtword();
private:
    int arts[NUM_ART];
    double sample_freq;
    // TODO: May want to set the seed for generator.
    std::default_random_engine generator;
    std::normal_distribution<double> hold_time;
    std::uniform_real_distribution<double> activation;
};

#endif /* RandomStim_h */
