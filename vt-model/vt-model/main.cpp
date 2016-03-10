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
    //Speaker male("Male",2);
    //cout << male.nose.Dx << "\n";
    Delta test(40);
    
    cout << test.tube->at(0).k1left1 << "\n";
    
    cout << "Hello, World!\n";
    return 0;
}
