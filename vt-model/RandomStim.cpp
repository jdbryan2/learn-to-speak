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
                       const std::uniform_real_distribution<double>::param_type activation_param) {
    // Articulators that we want to randomly stimulate
    for(int i=0; i<=kArt_muscle_MAX; i++)
        arts[i] = i;
    utterance_length = utterance_length_;
    sample_freq = sample_freq_;
    // std::normal_distribution<double>::param_type(0.2,0.25)
    hold_time.param(hold_time_param);
    //std::uniform_real_distribution<double>::param_type(0.0,1.0)
    activation.param(activation_param);
    rand_smooth.Init(utterance_length);
    //CreateArtword();
}

void RandomStim::CreateArtword() {
    double hold_times [kArt_muscle_MAX] = {0};
    int art = 0;
    for (double time = 0.0; time <= utterance_length; time = time + 1/sample_freq)
    {
        for (int i = 0; i < NUM_ART; i++)
        {
            art = arts[i];
            if (hold_times[art] <= 0.0 || (time+1/sample_freq) >= utterance_length) {
                rand_smooth.setTarget(art, time, activation(generator));
                hold_times[art] = hold_time(generator);
                continue;
            }
            hold_times[art] -= 1/sample_freq;
        }
    }
}

void RandomStim::NewArtword() {
    rand_smooth.resetTargets();
    CreateArtword();
}

void RandomStim::doControl(Speaker *speaker) {
    rand_smooth.intoArt(speaker->art, speaker->NowSeconds());
}

void RandomStim::InitArts(Speaker *speaker) {
    // Initializes articulator positions of speaker before simulation begins.
    // Necessary to avoid large discontinuites that make the simulation go unstable
    rand_smooth.intoArt(speaker->art, 0.0);
}
