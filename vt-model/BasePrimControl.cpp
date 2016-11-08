//
//  BasePrimControl.cpp
//  vt-model
//
//  Created by William J Wagner on 9/21/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "BasePrimControl.h"
#include <fstream>
#include <stdio.h>
#include "gsl/gsl_blas.h"

#define NUM_FEAT (MAX_NUMBER_OF_TUBES+kArt_muscle_MAX)


int BasePrimControl::LoadPrims() {
    std::string f_p_file("f_p_mat.prim");
    std::string num_prims_file("num_prim.prim");
    std::string area_std_file("area_std.prim");
    std::string art_std_file("art_std.prim");
    std::string samp_freq_file("samp_freq.prim");
    std::string mean_file("mean_mat.prim");
    std::string K_file("K_mat.prim");
    std::string O_file("O_mat.prim");
    std::string Oa_inv_file("Oa_inv_mat.prim");
    
    std::ifstream f_stream;
    FILE* f_stream_mat;
    std::string filename;
    
    filename = file_prefix + f_p_file;
    f_stream.open(filename);
    f_stream >> f_p[0];
    f_stream >> f_p[1];
    f_stream.close();
    
    filename = file_prefix + num_prims_file;
    f_stream.open(filename);
    f_stream >> num_prim;
    f_stream.close();
    
    filename = file_prefix + area_std_file;
    f_stream.open(filename);
    f_stream >> area_std;
    f_stream.close();
    
    filename = file_prefix + art_std_file;
    f_stream.open(filename);
    f_stream >> art_std;
    f_stream.close();
    
    filename = file_prefix + samp_freq_file;
    f_stream.open(filename);
    f_stream >> sample_freq;
    f_stream.close();
    
    double mean_len = (f_p[0]+f_p[1])*NUM_FEAT;
    filename = file_prefix + mean_file;
    f_stream_mat = fopen(filename.c_str(), "r");
    feat_mean = gsl_vector_alloc(mean_len);
    gsl_vector_fscanf(f_stream_mat, feat_mean);
    fclose(f_stream_mat);
    
    filename = file_prefix + K_file;
    f_stream_mat = fopen(filename.c_str(), "r");
    K = gsl_matrix_alloc(num_prim, f_p[1]*NUM_FEAT);
    gsl_matrix_fscanf(f_stream_mat, K);
    fclose(f_stream_mat);
    
    filename = file_prefix + O_file;
    f_stream_mat = fopen(filename.c_str(), "r");
    O = gsl_matrix_alloc(f_p[0]*NUM_FEAT,num_prim);
    gsl_matrix_fscanf(f_stream_mat, O);
    fclose(f_stream_mat);
    
    filename = file_prefix + Oa_inv_file;
    f_stream_mat = fopen(filename.c_str(), "r");
    Oa_inv = gsl_matrix_alloc(num_prim,f_p[0]*MAX_NUMBER_OF_TUBES);
    gsl_matrix_fscanf(f_stream_mat, Oa_inv);
    fclose(f_stream_mat);
    
    Yp = gsl_vector_alloc(f_p[1]*NUM_FEAT);
    Yp_unscaled = gsl_vector_alloc(f_p[1]*NUM_FEAT);
    Yf = gsl_vector_alloc(f_p[0]*NUM_FEAT);
    Yf_unscaled = gsl_vector_alloc(f_p[0]*NUM_FEAT);
    x_past = gsl_vector_alloc(num_prim);
    x = gsl_vector_alloc(num_prim);
    
    return 0;
}

BasePrimControl::BasePrimControl(double utterance_length_, int _control_period, Articulation initial_art, std::string prim_file_prefix,const gsl_vector * Aref_):
                Control(utterance_length_),
                control_period(_control_period)
{
    file_prefix = prim_file_prefix;
    LoadPrims();
    if (Aref_ != nullptr) {
        // Assume for now that Aref is sampled at log_freq
        int nsamp = Aref_->size;
        Aref = gsl_vector_alloc(nsamp);
        gsl_vector_memcpy(Aref, Aref_);
        Afref = gsl_vector_alloc(f_p[0]*MAX_NUMBER_OF_TUBES);
        doArefControl = true;
    }
    //TODO: Should do this with a constructor instead if Articualtion is made into a class
    for (int i=0; i<kArt_muscle_MAX; i++) {
        last_art[i] = initial_art[i];
    }
    
    // TESTING: Override initial_art with art generated by passing part of the mean through DFA
    // Should result in Yf_unscaled just being the future mean
    //gsl_vector_const_view past_mean = gsl_vector_const_subvector(feat_mean, (f_p[1]-1)*NUM_FEAT, NUM_FEAT);
    //gsl_vector_const_view past_mean = gsl_vector_const_subvector(feat_mean, 0, NUM_FEAT);
    //StepDFA(&past_mean.vector);
    /*for (int i=0; i<kArt_muscle_MAX; i++) {
        int ind = (f_p[1])*NUM_FEAT+MAX_NUMBER_OF_TUBES+i; // Future art mean 1
        last_art[i] = gsl_vector_get(feat_mean, ind);
    }*/
}

BasePrimControl::~BasePrimControl() {
    gsl_matrix_free(O);
    gsl_matrix_free(K);
    gsl_vector_free(feat_mean);
    gsl_matrix_free(Oa_inv);
    gsl_vector_free(Yp);
    gsl_vector_free(Yp_unscaled);
    gsl_vector_free(Yf);
    gsl_vector_free(Yf_unscaled);
    gsl_vector_free(x_past);
    gsl_vector_free(x);
    if (doArefControl) {
        gsl_vector_free(Aref);
        gsl_vector_free(Afref);
    }
    
}

void BasePrimControl::doControl(Speaker * speaker)
{
    // This statement sets the rate at which
    if( speaker->Now() % control_period == 0)
    {
        AreaFcn AreaFcn;
        speaker->getAreaFcn(AreaFcn);
        // Create feature vector of last area function and articulator activations
        gsl_vector *  feat = gsl_vector_alloc(NUM_FEAT);
        for (int i=0; i<NUM_FEAT; i++) {
            if (i<MAX_NUMBER_OF_TUBES) {
                gsl_vector_set(feat, i, AreaFcn[i]);
            }else {
                gsl_vector_set(feat, i, last_art[i-MAX_NUMBER_OF_TUBES]);
            }
        }
        // Setup Yp_unscaled to have a f_p[1] long constant history of initial_art and the initial VT area
        if (!isInitialized) {
            int ind;
            for( int i=0; i<f_p[1]; i++)
            {
                ind = i*(NUM_FEAT);
                gsl_vector_view Yp_u_i = gsl_vector_subvector(Yp_unscaled, ind, NUM_FEAT);
                gsl_vector_memcpy(&Yp_u_i.vector, feat);
            }
            // TESTING: Override Yp_unscaled with past_mean
            //gsl_vector_view past_mean = gsl_vector_subvector(feat_mean, 0, NUM_FEAT*f_p[1]);
            //gsl_vector_memcpy(Yp_unscaled, &past_mean.vector);
            if (doArefControl) {
                // Begin tracking starting at second sample because first sample is from IC's
                gsl_vector_view Afref_view = gsl_vector_subvector(Aref, MAX_NUMBER_OF_TUBES, f_p[0]*MAX_NUMBER_OF_TUBES);
                gsl_vector_memcpy(Afref, &Afref_view.vector);
            }
            isInitialized = true;
        }
        else
        {
            // Shift backward each feat sample by one in the vetor Yp_unscaled
            for( int i=0; i<=f_p[1]-2; i++)
            {
                // TODO check that the second parameter is an offset as expected from the docs.
                gsl_vector_view Yp_u_old = gsl_vector_subvector(Yp_unscaled, (i+1)*(NUM_FEAT), NUM_FEAT);
                gsl_vector_view Yp_u_older = gsl_vector_subvector(Yp_unscaled, i*(NUM_FEAT), NUM_FEAT);
                gsl_vector_memcpy(&Yp_u_older.vector, &Yp_u_old.vector);
            }
            // Store most recent feat vector in Yp
            gsl_vector_view Yp_u_recent = gsl_vector_subvector(Yp_unscaled, (f_p[1]-1)*(NUM_FEAT), NUM_FEAT);
            gsl_vector_memcpy(&Yp_u_recent.vector, feat);
            // Store next f future Area reference in Afref
            if (doArefControl) {
                // TODO: Fix this to rely on function from Speaker. This is ugly
                // First sammple of Aref is from IC's and second is taken care of by initialization of Afref so the first sample where we would need to shift is 3
                static int ind_s = 2;
                static int ind_e = f_p[0];
                static gsl_vector_view Aref_last_view = gsl_vector_subvector(Aref, Aref->size-MAX_NUMBER_OF_TUBES,MAX_NUMBER_OF_TUBES);
                if (ind_s*MAX_NUMBER_OF_TUBES+f_p[0]*MAX_NUMBER_OF_TUBES> Aref->size) {
                    ind_e--;
                    if (ind_e>0) {
                        gsl_vector_view Aref_view = gsl_vector_subvector(Aref, ind_s*MAX_NUMBER_OF_TUBES,ind_e*MAX_NUMBER_OF_TUBES);
                        gsl_vector_view Afref_e_view = gsl_vector_subvector(Afref, 0, ind_e*MAX_NUMBER_OF_TUBES);
                        gsl_vector_memcpy(&Afref_e_view.vector, &Aref_view.vector);
                    }
                    
                    // If we are simulating longer than the ref then just fill Afref with the last value of Aref
                    if (ind_e<=0){ind_e=0;}
                    // Fill the rest of Afref with the last value of Aref
                    for (int i=ind_e; i<f_p[0]; i++) {
                        gsl_vector_view Afref_i_view = gsl_vector_subvector(Afref, (i)*MAX_NUMBER_OF_TUBES, MAX_NUMBER_OF_TUBES);
                        gsl_vector_memcpy(&Afref_i_view.vector, &Aref_last_view.vector);
                    }
                }
                else {
                    gsl_vector_view Aref_view = gsl_vector_subvector(Aref, ind_s*MAX_NUMBER_OF_TUBES,f_p[0]*MAX_NUMBER_OF_TUBES);
                    gsl_vector_memcpy(Afref, &Aref_view.vector);
                }
                ind_s++;
            }
        }
        StepDFA(Yp_unscaled);

        // Set speaker's articulation
        for (int i = 0; i<kArt_muscle_MAX; i++) {
            speaker->art[i] = last_art[i];
        }

        gsl_vector_free(feat);
    }
}

void BasePrimControl::InitialArt(double *art) {
    for (int i = 0; i<kArt_muscle_MAX; i++) {
        art[i] = last_art[i];
    }
}

void BasePrimControl::StepDFA(const gsl_vector * Yp_unscaled_){
    // Remove mean from Yp
    gsl_vector_const_view past_mean = gsl_vector_const_subvector(feat_mean, 0, f_p[1]*NUM_FEAT);
    gsl_vector_memcpy(Yp, Yp_unscaled_);
    gsl_blas_daxpy(-1.0, &past_mean.vector, Yp);
    if (doArefControl) {
        // Remove mean from Afref
        gsl_vector * Af_mean = gsl_vector_alloc(f_p[0]*MAX_NUMBER_OF_TUBES);
        for (int j=0; j<f_p[0]; j++) {
            int ind = (f_p[1]+j)*NUM_FEAT;
            gsl_vector_view mean_Afj = gsl_vector_subvector(feat_mean, ind, MAX_NUMBER_OF_TUBES);
            gsl_vector_view Af_mean_j = gsl_vector_subvector(Af_mean, MAX_NUMBER_OF_TUBES*j, MAX_NUMBER_OF_TUBES);
            gsl_vector_memcpy(&Af_mean_j.vector, &mean_Afj.vector);
        }
        gsl_blas_daxpy(-1, Af_mean, Afref);
    }
    
    // Scale the Area and Articulatory features by their respective standard deviations
    double inv_area_std = 1/area_std;
    double inv_art_std = 1/art_std;
    for (int i=0; i<f_p[1]; i++) {
        int ind = i*(NUM_FEAT);
        gsl_vector_view Yp_area_i = gsl_vector_subvector(Yp, ind, MAX_NUMBER_OF_TUBES);
        gsl_vector_view Yp_art_i = gsl_vector_subvector(Yp, ind+MAX_NUMBER_OF_TUBES, kArt_muscle_MAX);
        gsl_blas_dscal(inv_area_std, &Yp_area_i.vector);
        gsl_blas_dscal(inv_art_std, &Yp_art_i.vector);
    }
    if (doArefControl) {
        // Scale Afref by its standard deviation
        gsl_blas_dscal(inv_area_std, Afref);
    }
    
    // Now actually use the primitives to find the future values
    gsl_blas_dgemv(CblasNoTrans, 1, K, Yp, 0, x_past);
    // TESTING: Disable all but one of the primitives.
    /*double xarr[8] = {};
    static double inc = 0;
    inc =  inc-.5;
    for (int i=0; i<num_prim; i++) {
     if (i==10)
     //continue;//   gsl_vector_set(x, i, 1);
         gsl_vector_set(x, i, gsl_vector_get(x, i)+inc);
     //gsl_vector_set(x, i, 0.0);
        xarr[i] = gsl_vector_get(x, i);
     } */
    if (doArefControl) {
        ArefControl();
    }
    else {
        // Don't modify x_past
        gsl_vector_memcpy(x, x_past);
    }
    gsl_blas_dgemv(CblasNoTrans, 1, O, x, 0, Yf);
    
    // Now rescale Yf
    for (int i=0; i<f_p[0]; i++) {
        int ind = i*(NUM_FEAT);
        gsl_vector_view Yf_area_i = gsl_vector_subvector(Yf, ind, MAX_NUMBER_OF_TUBES);
        gsl_vector_view Yf_art_i = gsl_vector_subvector(Yf, ind+MAX_NUMBER_OF_TUBES, kArt_muscle_MAX);
        gsl_blas_dscal(area_std, &Yf_area_i.vector);
        gsl_blas_dscal(art_std, &Yf_art_i.vector);
    }
    
    // Add back in the mean to Yf
    gsl_vector_const_view future_mean = gsl_vector_const_subvector(feat_mean, f_p[1]*NUM_FEAT, f_p[0]*NUM_FEAT);
    gsl_vector_memcpy(Yf_unscaled, Yf);
    gsl_blas_daxpy(1.0, &future_mean.vector, Yf_unscaled);
    
    // Set last_art to the new command for the next timestep
    // TESTING: Increment an individual articulator command
    //static double inc2 = .8;
    //inc2 = inc2-.01;
    for (int i=0; i<kArt_muscle_MAX; i++) {
        int ind = i+MAX_NUMBER_OF_TUBES;
        last_art[i] = gsl_vector_get(Yf_unscaled, ind);
        //last_art[0] = inc2;
        // Ensure that art is betwen 0 and 1
        if (last_art[i]>1) {
            printf("Art%d Command Out of Range: %g\n",i,last_art[i]);
            last_art[i] = 1;
        }
        else if (last_art[i]<0) {
            printf("Art%d Command Out of Range: %g\n",i,last_art[i]);
            last_art[i] = 0;
        }
    }
}

void BasePrimControl::ArefControl() {
    // 1sd Order Discrete PID controller taken from pg 9 of http://portal.ku.edu.tr/~cbasdogan/Courses/Robotics/projects/Discrete_PID.pdf
    // PID Gains
    const double Kp = 0.2;//0.1;//.2
    const double Ki = 0.5;//0.5;
    const double Kd = 0.02;//0.02;
    const double I_limit = Ki*5; // TODO: How do I set this value?
    const double Ts = 1/sample_freq;
    //const double Kp = 0.0;
    //const double Ki = 0.0;
    //const double Kd = 0.1;
    // Coefficients of Discrete PID Controller

    // Create and Initialize Past error and output terms
    static int iter = 0;
    gsl_vector * Ek = gsl_vector_alloc(num_prim);
    static gsl_vector * Ek1 = gsl_vector_calloc(num_prim);
    
    // Project Afref into low dim space
    gsl_blas_dgemv(CblasNoTrans, 1, Oa_inv, Afref, 0, x);
    // Find predicted error
    gsl_vector_memcpy(Ek, x);
    gsl_blas_daxpy(-1, x_past, Ek);
    // Contributing PID Terms
    // Proportional
    gsl_vector * P = gsl_vector_alloc(num_prim);
    gsl_vector_memcpy(P, Ek);
    gsl_blas_dscal(Kp, P);
    // Integral
    static gsl_vector * I1 = gsl_vector_calloc(num_prim);
    gsl_vector * I = gsl_vector_calloc(num_prim);
    gsl_vector_memcpy(I, Ek);
    gsl_blas_daxpy(1.0, Ek1, I);
    gsl_blas_dscal(Ki*Ts/2, I);
    gsl_blas_daxpy(1.0, I1, I);
    // Do Anti-Windup
    for (int i=0; i<num_prim; i++) {
        if (gsl_vector_get(I, i)>I_limit) {
            gsl_vector_set(I, i, I_limit);
        }
    }
    // Derivative
    gsl_vector * D = gsl_vector_alloc(num_prim);
    gsl_vector_memcpy(D, Ek);
    gsl_blas_daxpy(-1.0, Ek1, D);
    gsl_blas_dscal(Kd/Ts, D);
    // Sum them all up
    gsl_blas_dscal(0.0,x); // Zero x to begin with
    gsl_blas_daxpy(1.0, P, x);
    gsl_blas_daxpy(1.0, I, x);
    gsl_blas_daxpy(1.0, D, x);
    
    
    /*
     static double a = Kp;
     static double b = 0;
     static double c = 0;
     static double d = 0;
     // Just do proportional control for first 2 timesteps then do full PID
     // TODO: need to look at correct way to initialize pid
     if (iter==2){
     a = Kp + Ki*Ts/2 + Kd/Ts;
     b = -Kp + Ki*Ts/2 - 2*Kd/Ts;
     c = Kd/Ts;
     d = 1;
     }
     
     //double a = Kp + Ki*Ts/2 + Kd/Ts;
     //double b = -Kp + Ki*Ts/2 - 2*Kd/Ts;
     //double c = Kd/Ts;
     // Create and Initialize Past error and output terms
     gsl_vector * Ek = gsl_vector_alloc(num_prim);
     static gsl_vector * Ek1 = gsl_vector_calloc(num_prim);
     static gsl_vector * Ek2 = gsl_vector_calloc(num_prim);
     static gsl_vector * Uk1 = gsl_vector_calloc(num_prim);
     
     // Project Afref into low dim space
     gsl_blas_dgemv(CblasNoTrans, 1, Oa_inv, Afref, 0, x);
     // Find error
     gsl_vector_memcpy(Ek, x);
     gsl_blas_daxpy(-1, x_past, Ek);
     
     // Find Uk1 + aEk + bEk1 + cEk2
     // Use x as temp var doing addition
     gsl_blas_dscal(0.0,x); // Zero x to begin with
     gsl_blas_daxpy(d, Uk1, x);
     gsl_blas_daxpy(a, Ek, x);
     gsl_blas_daxpy(b, Ek1, x);
     gsl_blas_daxpy(c, Ek2, x);
     //gsl_blas_daxpy(b, Ek1, x);
     //gsl_blas_daxpy(c, Ek2, x);
     */
    
    // Update Past vectors
    gsl_vector_memcpy(Ek1, Ek);
    gsl_vector_memcpy(I1, I);
    
    iter++;
    
    // Override PID right now
    //gsl_vector_memcpy(x, Ek);
    //gsl_blas_dscal(Kp, x);
}