//
//  RandomStim.cpp
//  vt-model
//
//  Created by William J Wagner on 9/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "RandomStim.h"
#include "Articulation_enums.h"


RandomStim::RandomStim(double utterance_length_, double sample_freq_,
                       const std::normal_distribution<double>::param_type hold_time_param,
                       const std::uniform_real_distribution<double>::param_type activation_param) :
                    ArtwordControl(utterance_length_)
{
    // Articulators that we want to randomly stimulate
    for(int i=0; i<=kArt_muscle_MAX; i++)
        arts[i] = i;
    sample_freq = sample_freq_;
    hold_time.param(hold_time_param);
    activation.param(activation_param);
}

void RandomStim::CreateArtword() {
    double hold_times [kArt_muscle_MAX] = {0};
    int art = 0;
    for (double time = 0.0; time <= artword.totalTime; time = time + 1/sample_freq)
    {
        for (int i = 0; i < NUM_ART; i++)
        {
            art = arts[i];
            if (hold_times[art] <= 0.0 || (time+1/sample_freq) >= artword.totalTime) {
                artword.setTarget(art, time, activation(generator));
                hold_times[art] = hold_time(generator);
                continue;
            }
            hold_times[art] -= 1/sample_freq;
        }
    }
}

void RandomStim::NewArtword() {
    artword.resetTargets();
    CreateArtword();
}

void RandomStim::doControl(Speaker *speaker) {
    artword.intoArt(speaker->art, speaker->NowSeconds());
}

void RandomStim::InitialArt(Articulation art) {
    // Initializes articulator positions of speaker before simulation begins.
    // Necessary to avoid large discontinuites that make the simulation go unstable
    artword.intoArt(art, 0.0);
}
