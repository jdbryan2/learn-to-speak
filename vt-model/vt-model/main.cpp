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

    Artword_Speaker_to_Sound(&apa, &female, 22050, 25, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, nullptr, -1, &apa_sound);
    int err = apa_sound.play();
    err = apa_sound.play();
    err = apa_sound.play();
    
    std::cout << err << "\n";

    return 0;
}
