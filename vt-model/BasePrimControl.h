//
//  BasePrimControl.h
//  vt-model
//
//  Created by William J Wagner on 9/21/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef BasePrimControl_h
#define BasePrimControl_h

#include <string>
#include "Control.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

class BasePrimControl : public Control {
public:
    BasePrimControl(double utterance_length_, int _control_period, Articulation initial_art_, std::string prim_file_prefix);
    ~BasePrimControl();
    void doControl(Speaker * speaker);
    void InitialArt(Articulation art);
private:
    int LoadPrims();
public:
    std::string file_prefix;
private:
    // Primitive Parameters to Read in
    int f_p[2];
    int num_prim;
    double area_std;
    double art_std;
    double sample_freq;
    gsl_matrix * O;
    gsl_matrix * K;
    gsl_vector * feat_mean;
    
    // Variables in DFA
    gsl_vector * Yf;
    gsl_vector * Yf_unscaled;
    gsl_vector * Yp;
    gsl_vector * Yp_unscaled;
    gsl_vector * x;
    
    // Other Variables
    Articulation last_art;
    bool isInitialized = false;
    int control_period;
};

#endif /* BasePrimControl_h */
