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
    virtual void doControl(Speaker * speaker) = 0;
};

#endif /* Control_h */
