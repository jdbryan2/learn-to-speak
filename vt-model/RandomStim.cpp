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

RandomStim::RandomStim(double utterance_length, double sample_freq,
               const std::normal_distribution<double>::param_type hold_time_param,
               const std::uniform_real_distribution<double>::param_type activation_param,
               const std::uniform_real_distribution<double>::param_type lungs_exhale_param,
               const std::uniform_real_distribution<double>::param_type lungs_activation_param,
               const std::uniform_real_distribution<double>::param_type lungs_deactivation_param):
    RandomStim(utterance_length, sample_freq, hold_time_param, activation_param)
{
    
    std::uniform_real_distribution<double>::param_type lungs_inhale_param(0.3, 0.9); // inhale time

    lungs_inhale_time.param(lungs_inhale_param);
    lungs_exhale_time.param(lungs_exhale_param);

    lungs_activation.param(lungs_activation_param);
    lungs_deactivation.param(lungs_deactivation_param);

}

void RandomStim::CreateArtword() {
    double hold_times [kArt_muscle_MAX] = {0};
    int art = 0;
    for (double time = 0.0; time <= artword.totalTime; time = time + 1/sample_freq)
    {
        for (int i = 1; i < NUM_ART; i++) // skip lungs
        {
            art = arts[i];
            if (hold_times[art] <= 0.0 || (time+1/sample_freq) >= artword.totalTime) {
                artword.setTarget(art, time, activation(generator));
                hold_times[art] = hold_time(generator);
                if (hold_times[art] < 0.01) {
                    hold_times[art] = 0.01;
                }
                continue;
            }
            hold_times[art] -= 1/sample_freq;
        }
    }

    int breathe = 1;
    art = kArt_muscle_LUNGS;
    for (double time = 0.0; time <= artword.totalTime; time = time + 1/sample_freq)
    {
        if (hold_times[art] <= 0.0 || (time+1/sample_freq) >= artword.totalTime) {
            if(breathe == 1) {
                artword.setTarget(art, time, lungs_activation(generator));
                hold_times[art] = lungs_exhale_time(generator);
                breathe = -1;
            } else {
                artword.setTarget(art, time, lungs_deactivation(generator));
                hold_times[art] = lungs_inhale_time(generator);
                breathe = 1;
            }
            continue;
        }
        hold_times[art] -= 1/sample_freq;
    }
   /* // pump the lungs
    double time = 0.0;
    double hold = 0.0;
    while (time <= artword.totalTime) {
        // kArt_muscle_LUNGS
        artword.setTarget(kArt_muscle_LUNGS, time, lungs_activation(generator));
        time += lungs_inhale_time(generator);
        // exhale
        if (time < artword.totalTime) {
            artword.setTarget(kArt_muscle_LUNGS, time, lungs_deactivation(generator));
            time += lungs_exhale_time(generator);
            
        }
    } */
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
