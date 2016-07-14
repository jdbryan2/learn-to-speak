//
//  VocalTract.hpp
//  vt-model
//
//  Created by William J Wagner on 7/14/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#ifndef VocalTract_h
#define VocalTract_h

#include "Speaker_def.h"
#include "Delta.h"
#include "Articulation_enums.h"
#include "Sound.h"
#include <string>
#include <random>
#include <fstream>

#define DLIP  5e-3
#define ART_SPEAKER_MESHCOUNT 27
#define SMOOTH_LUNGS  true
#define FIRST_TUBE  6

class VocalTract {
public:
    VocalTract(std::string kindOfSpeaker, int numberOfVocalCordMasses);
protected:
    double MeshSumX(int i);
    double MeshSumY(int i);
    void MeshUpper (double art[kArt_muscle_MAX]);
private:
    void ArticulateUpper (double art[kArt_muscle_MAX]);
    double toLine (double x, double y, int i);
    int inside (double x, double y);
    static double arcLength (double from, double to);
    
protected:
    // ***** FIXED PARAMETERS FOR MERMELSTEIN'S MODEL ***** //
    double relativeSize;  // Relative size of the parameters. Different for female, male, child.
    // In the larynx.
    Speaker_CordDimensions cord;
    Speaker_CordSpring lowerCord;
    Speaker_CordSpring upperCord;
    Speaker_GlottalShunt shunt;
    // Above the larynx.
    Speaker_Velum velum;
    Speaker_Palate palate;
    Speaker_Tip tip;
    double neutralBodyDistance;
    Speaker_Alveoli alveoli;
    Speaker_TeethCavity teethCavity;
    Speaker_LowerTeeth lowerTeeth;
    Speaker_UpperTeeth upperTeeth;
    Speaker_Lip lowerLip;
    Speaker_Lip upperLip;
    // In the nasal cavity.
    Speaker_Nose nose;
    
    // ***** PARAMETERS FOR MESHING VOCAL TRACT ***** //
    // TODO: Move some of these into functions and make static variables.
    double xe [30] = {0.0}, ye [30] = {0.0}, xi [30] = {0.0}, yi [30] = {0.0},
    xmm [30] = {0.0}, ymm [30] = {0.0};
    int closed [40] = {0};
    
    double xm [40] = {0.0}, ym [40] = {0.0};
    
    double intX [1 + 16] = {0.0}, intY [1 + 16] = {0.0}, extX [1 + 11] = {0.0},
    extY [1 + 11] = {0.0};
    
    double bodyX = 0.0, bodyY = 0.0, bodyRadius= 0.0;
    
};

#endif /* VocalTract_h */
