//
//  Control.h
//  vt-model
//
//  Created by William J Wagner on 9/17/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef Control_h
#define Control_h

#include "Speaker.h"

class Control {
public:
    Control(double utterance_length_):utterance_length(utterance_length_){};
    virtual void doControl(Speaker * speaker) = 0;
    virtual void InitialArt(Articulation art) = 0;
public:
    double utterance_length;
};

#endif /* Control_h */
