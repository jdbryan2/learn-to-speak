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
    
    Yp = gsl_vector_alloc(f_p[1]*NUM_FEAT);
    Yp_unscaled = gsl_vector_alloc(f_p[1]*NUM_FEAT);
    Yf = gsl_vector_alloc(f_p[0]*NUM_FEAT);
    Yf_unscaled = gsl_vector_alloc(f_p[0]*NUM_FEAT);
    x = gsl_vector_alloc(num_prim);
    
    return 0;
}

BasePrimControl::BasePrimControl(double utterance_length_, int _control_period, Articulation initial_art_, std::string prim_file_prefix):
                Control(utterance_length_),
                control_period(_control_period)
{
    file_prefix = prim_file_prefix;
    LoadPrims();
    //TODO: Should do this with a constructor instead if Articualtion is made into a class
    for (int i=0; i<kArt_muscle_MAX; i++) {
        last_art[i] = initial_art_[i];
    }
}

BasePrimControl::~BasePrimControl() {
    gsl_matrix_free(O);
    gsl_matrix_free(K);
    gsl_vector_free(feat_mean);
    gsl_vector_free(Yp);
    gsl_vector_free(Yp_unscaled);
    gsl_vector_free(Yf);
    gsl_vector_free(Yf_unscaled);
    gsl_vector_free(x);
    
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
                // TODO check that the second parameter is an offset as expected from the docs.
                gsl_vector_view Yp_u_i = gsl_vector_subvector(Yp_unscaled, ind, NUM_FEAT);
                gsl_vector_memcpy(&Yp_u_i.vector, feat);
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
        }

        // Remove mean from Yp
        gsl_vector_const_view past_mean = gsl_vector_const_subvector(feat_mean, 0, f_p[1]*NUM_FEAT);
        gsl_vector_memcpy(Yp, Yp_unscaled);
        gsl_blas_daxpy(-1.0, &past_mean.vector, Yp);

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

        // Now actually use the primitives to find the future values
        gsl_blas_dgemv(CblasNoTrans, 1, K, Yp, 0, x);
        // TESTING: Disable all but one of the primitives.
         /*for (int i=0; i<num_prim; i++) {
            if (i==0)
                continue;//   gsl_vector_set(x, i, 1);
            gsl_vector_set(x, i, 0.0);
        }*/
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
        for (int i=0; i<kArt_muscle_MAX; i++) {
            int ind = i+MAX_NUMBER_OF_TUBES;
            last_art[i] = gsl_vector_get(Yf_unscaled, ind);
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