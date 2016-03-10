//
//  main.cpp
//  vt-model
//
//  Created by William J Wagner on 3/6/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include <iostream>
#include "Speaker.h"
#include "Delta.h"
#include "Articulation_enums.h"
#include "Articulation.h"
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    Speaker newspeaker("Male", 2);
    Delta newdelta(20);
    Art test;
    test.art[kArt_muscle_DEFAULT] = .67;
    

    std::cout << "Hello, World!\n";
    std::cout << newdelta.numberOfTubes;
    std::cout << "\n";
    std::cout << newspeaker.relativeSize;
    std::cout << "\n";
    std::cout << test.art[kArt_muscle_DEFAULT];
    std::cout << "\n";
    return 0;
}
