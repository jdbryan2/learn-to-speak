//
//  main.cpp
//  vt-model
//
//  Created by William J Wagner on 3/6/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

// #define NDEBUG 1 //disable assertions in the code
// TODO: Create include file and add assert.h. needs to be in only one spot.

#include <iostream>
#include <string>
#include "Speaker.h"
#include "Artword.h"
#include "Control.h"
#include "ArtwordControl.h"
#include "RandomStim.h"
#include "BasePrimControl.h"
#include <gsl/gsl_matrix.h>

#define NUM_ART 29

using namespace std;

Artword apa () {
    Artword apa(0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    apa.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    apa.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    apa.setTarget(kArt_muscle_LUNGS,0,0.2);
    apa.setTarget(kArt_muscle_LUNGS,0.1,0);
    apa.setTarget(kArt_muscle_MASSETER,0.25,0.7);
    apa.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.25,0.2);
    return apa;
}


Artword sigh () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_LUNGS, 0, 0.1 );
    articulation.setTarget(kArt_muscle_LUNGS, 0.1, 0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);
    return articulation;
}

Artword ejective () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_LUNGS, 0, 0.1 );
    articulation.setTarget(kArt_muscle_LUNGS, 0.1, 0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,1.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0.5,1.0);

    articulation.setTarget(kArt_muscle_MASSETER,0.0,-0.3);
    articulation.setTarget(kArt_muscle_MASSETER,0.5,-0.3);
    articulation.setTarget(kArt_muscle_HYOGLOSSUS,0.0,0.5);
    articulation.setTarget(kArt_muscle_HYOGLOSSUS,0.5,0.5);

    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.1,0.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.15,1.0);

    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.0,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.17,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.2,1.0);

    articulation.setTarget(kArt_muscle_STYLOHYOID,0.0,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.22,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.27,1.0);

    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.29,1.0);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.32,0.0);

    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.35,1.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.38,0.5);
    articulation.setTarget(kArt_muscle_INTERARYTENOID,0.5,0.5);

    articulation.setTarget(kArt_muscle_STYLOHYOID,0.35,1.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.38,0.0);
    articulation.setTarget(kArt_muscle_STYLOHYOID,0.5,0.0);
    return articulation;
}

// bilabial click (functional phonology pg 140)
Artword click () {
    Artword articulation(0.5);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.0,0.9);
    articulation.setTarget(kArt_muscle_STYLOGLOSSUS,0.5,0.9);

    articulation.setTarget(kArt_muscle_MASSETER,0.0,0.25);
    articulation.setTarget(kArt_muscle_MASSETER,0.2,0.25);
    articulation.setTarget(kArt_muscle_MASSETER,0.3,-0.25);
    articulation.setTarget(kArt_muscle_MASSETER,0.5,-0.25);

    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.0,0.75);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.2,0.75);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.3,0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,0.5,0.0);
    return articulation;
}

void simulate(Speaker* speaker, Control* controller) {
    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    Articulation art;
    controller->InitialArt(art);
    
    // initialize the simulation and tell it how many seconds to buffer
    speaker->InitSim(controller->utterance_length, art);
    
    cout << "Simulating...\n";
    
    while (speaker->NotDone())
    {
        controller->doControl(speaker);
        // generate the next acoustic sample
        speaker->IterateSim();
    }
    cout << "Done!\n";
}

void sim_artword(Speaker* speaker, Artword* artword)
    {
    ArtwordControl awcontrol(artword);
    simulate(speaker, &awcontrol);
    for(int i =0; i< 10; i++)
    {
        cout << speaker->result->z[100*i] << ", ";
    }
    cout << endl;
    
    // simple interface for playing back the sound that was generated
    int input =  0;// set to zero to test the speed of simulation.
    while (true)
    {
        cout << "Press (1) to play the sound or any key to quit.\n";
        std::cin.clear();
        cin >> input;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        if(input == 1) {
            speaker->Speak();
        } else {
            break;
        }
    }
}

void random_stim_trials(Speaker* speaker,double utterance_length, double log_period) {
    //std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test3Area/logs/");
    std::string prefix ("/home/jacob/Projects/learn-to-speak/data/");

    std::normal_distribution<double>::param_type hold_time_param(0.2,0.25);
    std::uniform_real_distribution<double>::param_type activation_param(0.0,1.0);

    std::uniform_real_distribution<double>::param_type lungs_exhale_param(2.0,3.0);
    //std::uniform_real_distribution<double>::param_type lungs_inhale_param(0.1,0.5);
    std::uniform_real_distribution<double>::param_type lungs_activation_param(0.1,0.3);
    std::uniform_real_distribution<double>::param_type lungs_deactivation_param(0.0,0.1);

    RandomStim rs(utterance_length, speaker->fsamp, hold_time_param, activation_param, lungs_exhale_param, lungs_activation_param, lungs_deactivation_param);
    for (int trial=1; trial <= 50; trial++)
    {
        // Generate a new random artword
        rs.NewArtword();
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "datalog" + to_string(trial)+ ".log",log_period);
        cout << "Trial " << trial << "\n";
        simulate(speaker, &rs);
        speaker->Speak();
        speaker->SaveSound(prefix + "sound" + to_string(trial) + ".log");
    }
}

void prim_control(Speaker* speaker,double utterance_length, double log_period) {
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test3Area/");
    Artword artw = apa();
    Articulation art = {};
    artw.intoArt(art, 0.0);
    BasePrimControl prim(utterance_length,log_period,art,prefix);
    // Initialize the data logger
    speaker->ConfigDataLogger(prefix + "prim_logs/primlog" + to_string(1)+ ".log",log_period);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(prefix + "prim_logs/sound" + to_string(1) + ".log");
}

int main()
{
    double sample_freq = 8000;
    int oversamp = 90;
    int number_of_glottal_masses = 2;
    Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
    
    double utterance_length = 6;
    double log_freq = 50;
    int log_period = 1; //floor(sample_freq/log_freq);
    random_stim_trials(&female,utterance_length,log_period);
    //prim_control(&female, utterance_length, log_period);
    //Artword artword = apa();
    //sim_artword(&female, &artword);
    return 0;
}
