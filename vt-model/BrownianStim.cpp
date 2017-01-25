//
//  RandomStim.cpp
//  vt-model
//
//  Created by William J Wagner on 9/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "BrownianStim.h"
#include "Articulation_enums.h"


BrownianStim::BrownianStim(double utterance_length_, double delta_,
                       double variance_) :
                    ArtwordControl(utterance_length_)
{
    // Articulators that we want to randomly stimulate
    for (int k=0; k < BROWN_ART; k++) {
        arts[k] = k;
    } 

    delta = delta_;
    variance = variance_;
    std::normal_distribution<double>::param_type walk_params(0,sqrt(delta*variance));
    walk.param(walk_params);
}


void BrownianStim::CreateArtword() {
    int art = 0;
    double time = 0.0;
    while( time < artword.totalTime)
    {
        time+=delta;

        // if we're over time we adjust the variance of the random walk for the final time step
        if(time > artword.totalTime) {
            std::normal_distribution<double>::param_type walk_params(0,sqrt((artword.totalTime-time+delta)*variance));
            walk.param(walk_params);

            // reset time to the final sample
            time = artword.totalTime;
        } 

        for(art=0; art<BROWN_ART;art++){
            double target;
            target = artword.getTarget(art, time-delta)+walk(generator);
            if(target < 0.0){
                target = -1.0*target;
            }
            if(target > 1.0){
                target = target-trunc(target);
            }
            
            artword.setTarget(art, time, target);
        }

    }


}

void BrownianStim::NewArtword() {
    artword.resetTargets();
    CreateArtword();
}

void BrownianStim::doControl(Speaker *speaker) {
    artword.intoArt(speaker->art, speaker->NowSeconds());
}

void BrownianStim::InitialArt(Articulation art) {
    // Initializes articulator positions of speaker before simulation begins.
    // Necessary to avoid large discontinuites that make the simulation go unstable
    artword.intoArt(art, 0.0);
}
