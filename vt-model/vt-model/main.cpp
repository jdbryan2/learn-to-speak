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
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    Speaker newspeaker("Male", 2);
    Delta newdelta(20);

    std::cout << "Hello, World!\n";
    std::cout << newdelta.numberOfTubes;
    std::cout << "\n";
    std::cout << newspeaker.relativeSize;
    std::cout << "\n";
    return 0;
}
