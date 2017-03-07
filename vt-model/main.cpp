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
#include "BrownianStim.h"
#include "BasePrimControl.h"
#include <gsl/gsl_matrix.h>

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

Artword unstable () {
    Artword articulation(1.0);
    articulation.setTarget(kArt_muscle_LUNGS, 0.0, 0.75309590858857711);
    articulation.setTarget(kArt_muscle_LUNGS, 0.27024999999998411, 0.32997949304765417);
    articulation.setTarget(kArt_muscle_LUNGS, 0.41074999999996864, 0.91598820268168613);
    articulation.setTarget(kArt_muscle_LUNGS, 0.68337500000002005, 0.13565701632217769);
    articulation.setTarget(kArt_muscle_LUNGS, 0.68350000000002009, 0.24923117402862585);
    articulation.setTarget(kArt_muscle_LUNGS, 0.84737500000007482, 0.59625697257499666);
    articulation.setTarget(kArt_muscle_LUNGS, 0.99987500000012575, 0.37804446385418006);
    articulation.setTarget(kArt_muscle_LUNGS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.0, 0.17365220998817027);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.25437499999998586, 0.52140695658586278);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.46299999999996289, 0.6105491714501905);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.605374999999994, 0.15589583879992749);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.78600000000005432, 0.71565150619968299);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.88225000000008647, 0.9038577297367334);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.97650000000011794, 0.44978455269129175);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.99987500000012575, 0.7445054620956737);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.000125, 0.91253099487046496);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.20962499999999076, 0.30351385810450587);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.40412499999996937, 0.75455631478337148);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.77612500000005102, 0.1822617596683829);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.94562500000010762, 0.67539781603278504);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.99987500000012575, 0.89368165812242473);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI,0,0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.000125, 0.31904929332933152);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.2111249999999906, 0.056208407054591562);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.31549999999997913, 0.060845406774190439);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.52524999999996724, 0.19650379298816162);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.63837500000000502, 0.9939777251549704);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.671250000000016, 0.69876139224334066);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.75875000000004522, 0.20685668492483289);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.99987500000012575, 0.04734473867241823);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.000125, 0.79262812955798245);
    articulation.setTarget(kArt_muscle_MASSETER, 0.08325000000000006, 0.55991893947184423);
    articulation.setTarget(kArt_muscle_MASSETER, 0.09462500000000007, 0.50417251459111856);
    articulation.setTarget(kArt_muscle_MASSETER, 0.34649999999997572, 0.81162849613087562);
    articulation.setTarget(kArt_muscle_MASSETER, 0.41162499999996854, 0.39456923513414943);
    articulation.setTarget(kArt_muscle_MASSETER, 0.425624999999967, 0.46455199093588245);
    articulation.setTarget(kArt_muscle_MASSETER, 0.55037499999997563, 0.80183344684177049);
    articulation.setTarget(kArt_muscle_MASSETER, 0.7916250000000562, 0.47311957327652882);
    articulation.setTarget(kArt_muscle_MASSETER, 0.99987500000012575, 0.49231054893179638);
    articulation.setTarget(kArt_muscle_MASSETER, 1.0, 0.0);
    return articulation;
}

Artword unstable2() {
    Artword articulation(1.0);
    articulation.setTarget(kArt_muscle_LUNGS, 0.0, 0.94356413403645789);
    articulation.setTarget(kArt_muscle_LUNGS, 0.28349999999999997, 0.59638202157935238);
    articulation.setTarget(kArt_muscle_LUNGS, 0.73150000000000004, 0.4216740203526364);
    articulation.setTarget(kArt_muscle_LUNGS, 0.86450000000000004, 0.83202332606107865);
    articulation.setTarget(kArt_muscle_LUNGS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LUNGS, 1.1486252384186544, 0.31418252832907284);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.0, 0.38691723014282153);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.23275000000000001, 0.44596362383548888);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.326625, 0.7850592968614053);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.56412499999999999, 0.00084446985690646986);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 0.754, 0.55044039782704146);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_INTERARYTENOID, 1.0471991097015776, 0.00096087265391242159);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.000125, 0.50331576028852842);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.087374999999999994, 0.023368495304847862);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.229625, 0.85784625600983089);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.58237499999999998, 0.15606742774860799);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.82662500000000005, 0.048803767918438108);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 0.97750000000000003, 0.24254444818873866);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_ORBICULARIS_ORIS,1.1908989766148974,0.99067956464389328);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.000125, 0.82281003726176805);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.218, 0.82425397138648449);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.35325000000000001, 0.24596988972707653);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.35425000000000001, 0.04692264798176024);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.76324999999999998, 0.89643497397253002);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 0.90774999999999995, 0.080667759034903224);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_LEVATOR_PALATINI, 1.3443127287411367, 0.97247717311943593);
    articulation.setTarget(kArt_muscle_MASSETER, 0.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 0.000125, 0.59886875483083235);
    articulation.setTarget(kArt_muscle_MASSETER, 0.121375, 0.68035917094972265);
    articulation.setTarget(kArt_muscle_MASSETER, 0.53212499999999996, 0.45806205408614242);
    articulation.setTarget(kArt_muscle_MASSETER, 0.62212500000000004, 0.63985934328622407);
    articulation.setTarget(kArt_muscle_MASSETER, 0.98050000000000003, 0.32019867329196977);
    articulation.setTarget(kArt_muscle_MASSETER, 1.0, 0.0);
    articulation.setTarget(kArt_muscle_MASSETER, 1.040003424289184, 0.06688279008427446);
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
    //int i =0;
    while (speaker->NotDone())
    {
        controller->doControl(speaker);
        // generate the next acoustic sample
        speaker->IterateSim();
        //i++;
        //if (i>5) 
            //break;
    }
    cout << "Done!\n";
}

void sim_artword(Speaker* speaker, Artword* artword, std::string artword_name,double log_period, std::string prefix)
    {
    cout << "Artword Starting----------\n";
    ArtwordControl awcontrol(artword);
    // Initialize the data logger
    prefix = prefix + "artword_logs/";
    speaker->ConfigDataLogger(prefix + artword_name + to_string(1) + ".log",log_period);
    simulate(speaker, &awcontrol);

    speaker->SaveSound(prefix + artword_name + "_sound" + to_string(1) + ".log");
    
    // simple interface for playing back the sound that was generated
    int input =  0;// set to zero to test the speed of simulation.
    while (true)
    {
        cout << "Press (1) to play the sound or any key to quit.\n";
        std::cin.clear();
        //cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> input;
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        if(input == 1) {
            speaker->Speak();
            input =0;
        } else {
            break;
        }
    }
    cout << "Artword Ending-------\n";
}

void random_stim_trials(Speaker* speaker,double utterance_length, double log_period, std::string prefix) 
{
    std::normal_distribution<double>::param_type hold_time_param(0.1,0.1);
    std::uniform_real_distribution<double>::param_type activation_param(0.0,1.0);
    RandomStim rs(utterance_length, speaker->fsamp, hold_time_param, activation_param);
    for (int trial=1; trial <= 100; trial++)
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

void brownian_stim_trials(Speaker* speaker, double utterance_length, double log_period, std::string prefix) 
{
    
    double delta, variance;
    delta = 0.05;
    variance = 0.15;
    BrownianStim bs(utterance_length, delta, variance);

    double chunk_size = utterance_length; //20; // seconds
    double num_chunks = std::ceil(utterance_length/chunk_size);

    // Generate a new random artword
    bs.NewArtword();

    // pass the articulator positions into the speaker BEFORE initializing the simulation
    // otherwise, we just get a strong discontinuity after the first instant
    Articulation art;
    bs.InitialArt(art);

    // initialize the simulation and tell it how many seconds to buffer
    speaker->InitSim(chunk_size, art);

    for (int trial=1; trial <= num_chunks; trial++)
    {
        // Initialize the data logger
        speaker->ConfigDataLogger(prefix + "logs/datalog" + to_string(trial)+ ".log",log_period);
        cout << "Trial " << trial << " of " << num_chunks << "\n";
         
        speaker->InitDataLogger();
        
        cout << "Simulating...\n";
        
        while (speaker->LoopBack())
        {
            bs.doControl(speaker);
            // generate the next acoustic sample
            speaker->IterateSim();
        }
        speaker->Speak();
        speaker->SaveSound(prefix + "logs/sound" + to_string(trial) + ".log");
        
    }
    return;
}

void prim_control(Speaker* speaker,double utterance_length, double log_period, std::string prefix) {
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

void AreaRefControl(Speaker* speaker, double log_freq, double log_period, std::string prefix) {
    Artword artw = apa();
    Articulation art = {};
    artw.intoArt(art, 0.0);
    double utterance_length = artw.totalTime;
    
    std::ifstream f_stream;
    FILE* f_stream_mat;
    std::string filename;
    filename = prefix + "Aref.alog";
    f_stream_mat = fopen(filename.c_str(), "r");
    gsl_vector * Aref = gsl_vector_alloc(MAX_NUMBER_OF_TUBES*(utterance_length*log_freq+1));
    gsl_vector_fscanf(f_stream_mat, Aref);
    fclose(f_stream_mat);
    
    // Make control longer than sample
    utterance_length+=2,0;
    BasePrimControl prim(utterance_length,log_period,art,prefix,Aref);
    // Initialize the data logger
    speaker->ConfigDataLogger(prefix + "prim_logs/Areflog" + to_string(1)+ ".log",log_period);
    simulate(speaker, &prim);
    speaker->Speak();
    speaker->SaveSound(prefix + "prim_logs/Arefsound" + to_string(1) + ".log");
}

int main()
{
    double sample_freq = 8000;
    int oversamp = 80;
    int number_of_glottal_masses = 2;
    Speaker female("Female",number_of_glottal_masses, sample_freq, oversamp);
    //std::string prefix ("/Users/JacobWagner/Documents/Repositories/learn-to-speak/analysis/test5/");
    std::string prefix ("/home/jacob/Projects/learn-to-speak/data/");
    //std::string prefix ("/home/jacob/Projects/learn-to-speak/analysis/test3/");
    double utterance_length = 0.5;
    double desired_log_freq = 50;
    int log_period = floor(sample_freq/desired_log_freq);
    double log_freq = sample_freq/log_period;
    // 1.) Create Artword to track
    Artword artword = apa();
    std::string artword_name = "apa2";
    //Artword artword = unstable2();
    //std::string artword_name = "unstable2_artword";
    sim_artword(&female, &artword,artword_name,log_period,prefix);
    
    // 2.) Generate Randomly Stimulated data trials
    //random_stim_trials(&female,utterance_length,log_period,prefix);
    //brownian_stim_trials(&female,utterance_length,log_period,prefix);
    
    // 3.) Perform MATLAB DFA to find primitives and generate Aref of 1.)
    
    // 4.) Perform Primitive Control based on IC only
    //prim_control(&female, utterance_length, log_period,prefix);
    
    // 5.) Perform Area Function Tracking of 1.)
    //AreaRefControl(&female, log_freq, log_period,prefix);
    
    return 0;
}
