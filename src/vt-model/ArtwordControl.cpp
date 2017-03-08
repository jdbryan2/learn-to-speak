//
//  ArtwordControl.cpp
//  vt-model
//
//  Created by William J Wagner on 9/19/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "ArtwordControl.h"

ArtwordControl::ArtwordControl(double utterance_length_):
                Control(utterance_length_),
                artword(utterance_length_)
{
}

ArtwordControl::ArtwordControl(Artword* artword_):
                Control(artword_->totalTime)
{
    artword_->Copy(&artword);
}

void ArtwordControl::doControl(Speaker *speaker) {
    artword.intoArt(speaker->art, speaker->NowSeconds());
}

void ArtwordControl::InitialArt(Articulation art) {
    // Initializes articulator positions of speaker before simulation begins.
    // Necessary to avoid large discontinuites that make the simulation go unstable
    artword.intoArt(art, 0.0);
}