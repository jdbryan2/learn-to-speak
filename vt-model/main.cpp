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

void random_stim_trials(Speaker* speaker,double utterance_length, double log_freq) {
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test3Area/logs/");
    std::normal_distribution<double>::param_type hold_time_param(0.2,0.25);
    std::uniform_real_distribution<double>::param_type activation_param(0.0,1.0);
    RandomStim rs(utterance_length, speaker->fsamp, hold_time_param, activation_param);
    for (int trial=1; trial <= 5; trial++)
    {
        // Generate a new random artword
        rs.NewArtword();
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "datalog" + to_string(trial)+ ".log",log_freq);
        cout << "Trial " << trial << "\n";
        simulate(speaker, &rs);
        speaker->Speak();
        speaker->SaveSound(prefix + "sound" + to_string(trial) + ".log");
    }
}

void test_gsl_matrix () {
    int i, j;
    FILE* f_stream = fopen("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/mm_mat.txt","r");
    gsl_matrix * m = gsl_matrix_alloc(3,4);
    gsl_matrix_fscanf(f_stream, m);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 4; j++)
            printf ("m(%d,%d) = %g\n", i, j,
                    gsl_matrix_get (m, i, j));
    
    
    
    /*int i, j;
    FILE* f_stream = fopen("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test4mat/matrix1.log","w");
    gsl_matrix * m = gsl_matrix_alloc (10, 3);
    
    for (i = 0; i < 10; i++)
        for (j = 0; j < 3; j++)
            gsl_matrix_set (m, i, j, 0.23 + 100*i + j);
    
    gsl_matrix_fprintf(f_stream, m, "%f"); */
    
    /*for (i = 0; i < 100; i++)  // OUT OF RANGE ERROR
        for (j = 0; j < 3; j++)
            printf ("m(%d,%d) = %g\n", i, j,
                    gsl_matrix_get (m, i, j));
    */
    
    gsl_matrix_free (m);
}


void prim_control(Speaker* speaker,double utterance_length, double log_freq) {
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test3Area/");
    Articulation art = {};
    BasePrimControl prim(utterance_length,art,prefix);
    // Initialize the data logger
    speaker->ConfigDataLogger(prefix + "primlog/datalog" + to_string(1)+ ".log",log_freq);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(prefix + "sound" + to_string(1) + ".log");
}

int main()
{
    double sample_freq = 8000;
    int oversamp = 70;
    int number_of_glottal_masses = 2;
    Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
    
    double utterance_length = 4;
    double log_freq = 50;
    //random_stim_trials(&female,utterance_length,log_freq);
    prim_control(&female, utterance_length, log_freq);
    //Artword artword = apa();
    //sim_artword(&female, &artword);
    //test_gsl_matrix();
    return 0;
}
