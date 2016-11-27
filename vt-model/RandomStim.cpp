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
    //for(int i=0; i<=kArt_muscle_MAX; i++)
    //    arts[i] = i;
    arts[0] = kArt_muscle_INTERARYTENOID;
    arts[1] = kArt_muscle_LEVATOR_PALATINI;
    arts[2] = kArt_muscle_LUNGS;
    arts[3] = kArt_muscle_MASSETER;
    arts[4] = kArt_muscle_ORBICULARIS_ORIS;
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
    double hold_times [NUM_ART];
    int art = 0;
    double time = 0.0;
    for (int ind = 0; time < artword.totalTime; ind++)
    {
        // TODO: Maybe change from sample_freq to log_freq
        time = ind/sample_freq;
        for (int i = 0; i < NUM_ART; i++)
        {
            art = arts[i];
            if (hold_times[art] <= 0.0 || ind == 0) {
                artword.setTarget(art, time, activation(generator));
                hold_times[art] = hold_time(generator);
                if (hold_times[art] < 0.01) {
                    hold_times[art] = 0.01;
                }
                continue;
            }
            // Set last art target from default to same as last random generated one
            // TODO: Decide if we want to do this or to generate another random sample...
            else if (time >= artword.totalTime) {
                // What we were doing before
                //artword.setTarget(art, time, activation(generator));
                //hold_times[art] = hold_time(generator);
                //continue;
                
                // Another method. Don't interpolate, just set to constant of previous activation
                //double last_tar = artword.data[art].targets[artword.data[art].numberOfTargets-2].target_value;
                //artword.setTarget(art, time, last_tar);
                
                // Set target past end of artword so that articulations can be interpolated up til end of artword
                artword.setTarget(art, time+hold_times[art], activation(generator));
                // hold_times[art] won't be used again for this art because this is the last iteration of ind for loop
            }
            hold_times[art] -= 1/sample_freq;
        }
    }

    /*int breathe = 1;
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
    */
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
