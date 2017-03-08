//
//  ArtwordControl.hpp
//  vt-model
//
//  Created by William J Wagner on 9/19/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef ArtwordControl_h
#define ArtwordControl_h

#include "Control.h"
#include "Artword.h"


class ArtwordControl : public Control {
public:
    ArtwordControl(double utterance_length_);
    ArtwordControl(Artword* artword_);
    ~ArtwordControl() { }
    void doControl(Speaker* speaker);
    void InitialArt(Articulation art);
public:
    Artword artword;
};


#endif /* ArtwordControl_h */
