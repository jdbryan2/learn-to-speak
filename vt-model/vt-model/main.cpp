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
#include "Speaker_to_Delta.h"
#include "Art_Speaker_Delta.h"
#include "Artword.h"
#include "Artword_Speaker_to_Sound.h"

using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...
    Speaker newspeaker("Male", 10);
    Speaker female("Female",2);
    Artword apa(0.5);
    Sound apa_sound;
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    Delta newdelta1, newdelta2;
    Speaker_to_Delta(newspeaker, newdelta1);
    Artword artword_test(1.0);
    Art test;
    artword_test.intoArt(test, 0.5);
    test.art[kArt_muscle_LUNGS] = 0.67;
    Art_Speaker_intoDelta(test, newspeaker, newdelta2);
    
    Artword_Speaker_to_Sound(&apa, &female, 44100, 1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, &apa_sound);
    int err = apa_sound.play();
    
    std::cout << err << "\n";

    std::cout << "Hello, World!\n";
    std::cout << newdelta1.numberOfTubes;
    std::cout << "\n";
    std::cout << newspeaker.relativeSize;
    std::cout << "\n";
    std::cout << test.art[kArt_muscle_DEFAULT];
    std::cout << "\n";
    std::cout << newdelta1.tube[6].Dyeq;
    std::cout << "\n";
    std::cout << newdelta2.tube[6].Dyeq;
    std::cout << "\n";
    return 0;
}
