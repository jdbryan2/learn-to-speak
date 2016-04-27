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
//#include "Delta.h"
//#include "Articulation_enums.h"
//#include "Articulation.h"
//#include "Speaker_to_Delta.h"
//#include "Art_Speaker_Delta.h"
#include "Artword.h"
//#include "Artword_Speaker_to_Sound.h"

using namespace std;

int main(int argc, const char * argv[]) 
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

    for(int oversamp = 70; oversamp<71; oversamp+=10) {
                        // speaker type, number of glotal masses, fsamp, oversamp
        //Speaker female("Female",2, 22050, 25);
        Speaker female("Female",2, 8000, oversamp);

        // pass the articulator positions into the speaker BEFORE initializing the simulation
        // otherwise, we just get a strong discontinuity after the first instant
        apa.intoArt(female.art, 0.0);

        // initialize the simulation and tell it how many seconds to buffer
        female.InitSim(0.5);

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
        for(int i =0; i< 10; i++) {
            cout << female.result->z[100*i] << ", ";
        }
        cout << endl;
        //cout << female.Speak() << endl;

        // simple interface for playing back the sound that was generated
        int input =  0;// set to zero to test the speed of simulation.
        while (true){
            cout << "Press (1) to play the sound or any key to quit.\n";
            cin >> input;
            if(input == 1) {
                cout << female.Speak() << endl;
            } else {
                break;
            }

        }

    /*Artword apa(0.5);
    Sound apa_sound;
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);

    Artword_Speaker_to_Sound(&apa, &female, 22050, 25, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, &apa_sound);
    int err = apa_sound.play();
    err = apa_sound.play();
    err = apa_sound.play();
    
    std::cout << err << "\n";*/
    }
    return 0;
}
