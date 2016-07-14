//
//  main.cpp
//  vt-model
//
//  Created by William J Wagner on 3/6/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#define NDEBUG 1 //disable assertions in the code

#include <iostream>
#include "Speaker.h"
#include "Artword.h"

using namespace std;

int main()
{
    Artword apa(0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);

    int oversamp = 70;
    // speaker type, number of glotal masses, fsamp, oversamp
    Speaker female("Female",2, 8000, oversamp);

    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    apa.intoArt(female.art, 0.0);

    // initialize the simulation and tell it how many seconds to buffer
    female.InitSim(0.5, std::string ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/vt-model/vt-model/logs/datalog1.log"),800.0);

    cout << "Simulating...\n";
    cout << "Oversample rate = " << oversamp << endl;
    while (female.NotDone())
    {
        // adjust articulators using controller
        // Artword class is being used for this currently
        apa.intoArt(female.art, female.NowSeconds());

        // generate the next acoustic sample
        female.IterateSim();
    }
    cout << "Done!\n";
    for(int i =0; i< 10; i++)
    {
        cout << female.result->z[100*i] << ", ";
    }
    cout << endl;

    // simple interface for playing back the sound that was generated
    int input =  0;// set to zero to test the speed of simulation.
    while (true)
    {
        cout << "Press (1) to play the sound or any key to quit.\n";
        cin >> input;
        if(input == 1) {
            cout << female.Speak() << endl;
        } else {
            break;
        }
    }
    female.SaveSound(std::string ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/vt-model/vt-model/logs/recorded1.log"));
    return 0;
}
