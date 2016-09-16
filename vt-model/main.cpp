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

void sim_artword( Artword articulation, double utterance_length)
    {
    double sample_freq = 8000;
    int oversamp = 70;
    int number_of_glottal_masses = 2;
    
    int input2 =  0;// set to zero to test the speed of simulation.
    
    // speaker type, number of glotal masses, fsamp, oversamp
    Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
    
    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    articulation.intoArt(female.art, 0.0);
    
    // initialize the simulation and tell it how many seconds to buffer
    female.InitSim(utterance_length);
    
    cout << "Simulating. " << "\n";
    
    while (female.NotDone())
    {
        // adjust articulators using controller
        // Artword class is being used for this currently.
        // Could use feedback instead
        articulation.intoArt(female.art, female.NowSeconds());
        
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
    input2 =  0;// set to zero to test the speed of simulation.
    while (true)
    {
        cout << "Press (1) to play the sound or any key to quit.\n";
        std::cin.clear();
        cin >> input2;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        if(input2 == 1) {
            female.Speak();
        } else {
            break;
        }
    }
}

void random_stim() {
    std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test3Area/logs/");
    std::default_random_engine generator;
    std::normal_distribution<double> hold_time(0.2,0.25);
    std::uniform_real_distribution<double> activation(0.0,1.0);
    double utterance_length = 0.5;
    //Artword rand_smooth(utterance_length);
    double sample_freq = 8000;
    int oversamp = 70;
    int number_of_glottal_masses = 2;
    
    // Articulators that we want to randomly stimulate
    //int arts[NUM_ART] = {kArt_muscle_INTERARYTENOID,kArt_muscle_LEVATOR_PALATINI,kArt_muscle_MASSETER,kArt_muscle_ORBICULARIS_ORIS};
    
    int arts[NUM_ART] = {
        kArt_muscle_LUNGS,
        kArt_muscle_INTERARYTENOID,
        kArt_muscle_CRICOTHYROID,
        kArt_muscle_VOCALIS,
        kArt_muscle_THYROARYTENOID,
        kArt_muscle_POSTERIOR_CRICOARYTENOID,
        kArt_muscle_LATERAL_CRICOARYTENOID,
        kArt_muscle_STYLOHYOID,
        kArt_muscle_STERNOHYOID,
        kArt_muscle_THYROPHARYNGEUS,
        kArt_muscle_LOWER_CONSTRICTOR,
        kArt_muscle_MIDDLE_CONSTRICTOR,
        kArt_muscle_UPPER_CONSTRICTOR,
        kArt_muscle_SPHINCTER,
        kArt_muscle_HYOGLOSSUS,
        kArt_muscle_STYLOGLOSSUS,
        kArt_muscle_GENIOGLOSSUS,
        kArt_muscle_UPPER_TONGUE,
        kArt_muscle_LOWER_TONGUE,
        kArt_muscle_TRANSVERSE_TONGUE,
        kArt_muscle_VERTICAL_TONGUE,
        kArt_muscle_RISORIUS,
        kArt_muscle_ORBICULARIS_ORIS,
        kArt_muscle_LEVATOR_PALATINI,
        kArt_muscle_TENSOR_PALATINI,
        kArt_muscle_MASSETER,
        kArt_muscle_MYLOHYOID,
        kArt_muscle_LATERAL_PTERYGOID,
        kArt_muscle_BUCCINATOR};
    
    for (int trial=1; trial <= 30; trial++)
    {
        Artword rand_smooth(utterance_length);
        //rand_smooth.setTarget(kArt_muscle_LUNGS,0,0.2);
        //rand_smooth.setTarget(kArt_muscle_LUNGS,0.1,0);
        double hold_times [kArt_muscle_MAX] = {.25};
        int art = 0;
        
        for (double time = 0.0; time <= utterance_length; time = time + 1/sample_freq)
        {
            //for (int art = kArt_muscle_MIN; art < kArt_muscle_MAX; art++)
            for (int i = 0; i < NUM_ART; i++)
            {
                art = arts[i];
                if (hold_times[art] <= 0.0 || (time+1/sample_freq) >= utterance_length) {
                    rand_smooth.setTarget(art, time, activation(generator));
                    hold_times[art] = hold_time(generator);
                    continue;
                }
                hold_times[art] -= 1/sample_freq;
            }
        }
        
        // speaker type, number of glotal masses, fsamp, oversamp
        Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
        
        // pass the articulator positions into the speaker BEFORE initializing the simulation
        // otherwise, we just get a strong discontinuity after the first instant
        rand_smooth.intoArt(female.art, 0.0);
        
        // initialize the simulation and tell it how many seconds to buffer
        female.InitSim(0.5, prefix + "datalog" + to_string(trial)+ ".log",50.0);
        
        cout << "Simulating. Trial " << trial << "\n";
        
        while (female.NotDone())
        {
            // adjust articulators using controller
            // Artword class is being used for this currently.
            // Could use feedback instead
            rand_smooth.intoArt(female.art, female.NowSeconds());
            
            // generate the next acoustic sample
            female.IterateSim();
        }
        cout << "Done!\n";
        female.Speak();
        // TODO: Figure out why  SaveSound() breaks on linux
        // Comment out to run on Linux
        female.SaveSound(prefix + "sound" + to_string(trial) + ".log");
    }
}

void test_gsl_matrix () {
    int i, j;
    FILE* f_stream = fopen("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test4mat/matrix1.log","w");
    gsl_matrix * m = gsl_matrix_alloc (10, 3);
    
    for (i = 0; i < 10; i++)
        for (j = 0; j < 3; j++)
            gsl_matrix_set (m, i, j, 0.23 + 100*i + j);
    
    gsl_matrix_fprintf(f_stream, m, "%f");
    
    /*for (i = 0; i < 100; i++)  // OUT OF RANGE ERROR
        for (j = 0; j < 3; j++)
            printf ("m(%d,%d) = %g\n", i, j,
                    gsl_matrix_get (m, i, j));
    */
    
    gsl_matrix_free (m);
}


int main()
{
    sim_artword(click(), 0.5);
    test_gsl_matrix();
    return 0;
}
