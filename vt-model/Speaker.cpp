/* Speaker.cpp
 *
 * Copyright (C) 1992-2011,2015 Paul Boersma
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

#include "Speaker.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>

// ***** Constants for modifying acoustic simulation
#define DYMIN  0.00001
#define CRITICAL_VELOCITY  10.0
#define NOISE_FACTOR  0.1

// While debugging, some of these can be 1; otherwise, they are all 0: 
#define EQUAL_TUBE_WIDTHS  0
#define CONSTANT_TUBE_LENGTHS  0 // was set to 1
#define NO_MOVING_WALLS  0
#define NO_TURBULENCE  0 // NOTE: Settting to 0 will cause a given artword to produce a different sound when called multiple times with the same speaker.
#define NO_RADIATION_DAMPING  0
#define NO_BERNOULLI_EFFECT  0
#define MASS_LEAPFROG  0
#define B91  0
// ***** End acoustic simulation constants

using namespace std;

Speaker::Speaker(string kindOfSpeaker, int numberOfVocalCordMasses, double samplefreq, int oversample_multiplier):
    VocalTract(kindOfSpeaker,numberOfVocalCordMasses),
    Delta(),
    fsamp(samplefreq),
    oversamp(oversample_multiplier),
    distribution(1.0, NOISE_FACTOR)
{
    InitializeTube();
    result = new Sound();
}

void Speaker::InitializeTube()
{
    // Map speaker parameters into delta tube
    // It is only necessary to call this once for each speaker.
    // It does not need to be called for a new articulation with the same speaker
    double f = relativeSize * 1e-3;   // we shall use millimetres and grams
    int itube;
    assert(cord.numberOfMasses == 1 || cord.numberOfMasses == 2 || cord.numberOfMasses == 10);
    assert(numberOfTubes == 89);
    
    // Lungs: tubes 0..22.
    for (itube = 0; itube <= 22; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> Dx = t -> Dxeq = 10.0 * f;
        t -> Dy = t -> Dyeq = 100.0 * f;
        t -> Dz = t -> Dzeq = 230.0 * f;
        t -> mass = 10.0 * relativeSize * t -> Dx * t -> Dz;   // 80 * f; 35 * Dx * Dz
        t -> k1 = 200.0;   // 90000 * Dx * Dz; Newtons per metre
        t -> k3 = 0.0;
        t -> Brel = 0.8;
        t -> parallel = 1000;
    }
    
    // Bronchi: tubes 23..28.
    
    for (itube = 23; itube <= 28; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> Dx = t -> Dxeq = 10.0 * f;
        t -> Dy = t -> Dyeq = 15.0 * f;
        t -> Dz = t -> Dzeq = 30.0 * f;
        t -> mass = 10.0 * f;
        t -> k1 = 40.0;   // 125000 * Dx * Dz; Newtons per metre
        t -> k3 = 0.0;
        t -> Brel = 0.8;
    }
    
    // Trachea: tubes 29..34; four of these may be replaced by conus elasticus (see below).
    
    for (itube = 29; itube <= 34; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> Dx = t -> Dxeq = 10.0 * f;
        t -> Dy = t -> Dyeq = 15.0 * f;
        t -> Dz = t -> Dzeq = 16.0 * f;
        t -> mass = 5.0 * f;
        t -> k1 = 160.0;   // 100000 * Dx * Dz; Newtons per metre
        t -> k3 = 0.0;
        t -> Brel = 0.8;
    }
    
    if (SMOOTH_LUNGS) {
        struct { int itube; double Dy, Dz, parallel; } data [] = {
            {  6, 120.0, 240.0, 5000.0 }, {  7, 120.0, 240.0, 5000.0 }, {  8, 120.0, 240.0, 5000.0 },
            {  9, 120.0, 240.0, 5000.0 }, { 10, 120.0, 240.0, 5000.0 }, { 11, 120.0, 240.0, 5000.0 },
            { 12, 120.0, 240.0, 2500.0 }, { 13, 120.0, 240.0, 1250.0 }, { 14, 120.0, 240.0,  640.0 },
            { 15, 120.0, 240.0,  320.0 }, { 16, 120.0, 240.0,  160.0 }, { 17, 120.0, 140.0,   80.0 },
            { 18,  70.0,  70.0,   40.0 }, { 19,  35.0,  35.0,   20.0 }, { 20,  18.0,  18.0,   10.0 },
            { 21,  12.0,  12.0,    5.0 }, { 22,  12.0,  12.0,    3.0 }, { 23,  18.0,   9.0,    2.0 },
            { 24,  18.0,  19.0,    2.0 }, { 0 } };
        int i;
        for (i = 0; data [i]. itube; i ++) {
            Delta_Tube t = &(tube[data[i].itube]);
            t -> Dy = t -> Dyeq = data [i]. Dy * f;
            t -> Dz = t -> Dzeq = data [i]. Dz * f;
            t -> parallel = data [i]. parallel;
        }
        for (itube = 25; itube <= 34; itube ++) {
            Delta_Tube t = &(tube[itube]);
            t -> Dy = t -> Dyeq = 11.0 * f;
            t -> Dz = t -> Dzeq = 14.0 * f;
            t -> parallel = 1;
        }
        for (itube = FIRST_TUBE; itube <= 17; itube ++) {
            Delta_Tube t = &(tube[itube]);
            t -> Dx = t -> Dxeq = 10.0 * f;
            t -> mass = 10.0 * relativeSize * t -> Dx * t -> Dz;   // 10 mm
            t -> k1 = 1e5 * t -> Dx * t -> Dz;   // elastic tissue: 1 mbar/mm
            t -> k3 = 0.0;
            t -> Brel = 1.0;
        }
        for (itube = 18; itube <= 34; itube ++) {
            Delta_Tube t = &(tube[itube]);
            t -> Dx = t -> Dxeq = 10.0 * f;
            t -> mass = 3.0 * relativeSize * t -> Dx * t -> Dz;   // 3 mm
            t -> k1 = 10e5 * t -> Dx * t -> Dz;   // cartilage: 10 mbar/mm
            t -> k3 = 0.0;
            t -> Brel = 1.0;
        }
    }
    
    // Glottis: tubes 35 and 36; the last one may be disconnected (see below).
    {
        Delta_Tube t = &(tube[35]);
        t -> Dx = t -> Dxeq = lowerCord.thickness;
        t -> Dy = t -> Dyeq = 0.0;
        t -> Dz = t -> Dzeq = cord.length;
        t -> mass = lowerCord.mass;
        t -> k1 = lowerCord.k1;
        t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
        t -> Brel = 0.2;
    }
    
    // Fill in the values for the upper part of the glottis (tube 36) only if there is no one-mass model.
    if (cord.numberOfMasses >= 2) {
        Delta_Tube t = &(tube[36]);
        t -> Dx = t -> Dxeq = upperCord.thickness;
        t -> Dy = t -> Dyeq = 0.0;
        t -> Dz = t -> Dzeq = cord.length;
        t -> mass = upperCord.mass;
        t -> k1 = upperCord.k1;
        t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
        t -> Brel = 0.2;
        
        // Couple spring with lower cord.
        t -> k1left1 = tube[35].k1right1 = 1.0;
    }
    
    // Fill in the values for the conus elasticus (tubes 78..85) only if we want to model it.
    if (cord.numberOfMasses == 10) {
        tube[78].Dx = tube[78]. Dxeq = 8.0 * f;
        tube[79].Dx = tube[79].Dxeq = 7.0 * f;
        tube[80].Dx = tube[80].Dxeq = 6.0 * f;
        tube[81].Dx = tube[81].Dxeq = 5.0 * f;
        tube[82].Dx = tube[82].Dxeq = 4.0 * f;
        tube[83].Dx = tube[83].Dxeq = 0.75 * 4.0 * f + 0.25 * lowerCord.thickness;
        tube[84].Dx = tube[84].Dxeq = 0.50 * 4.0 * f + 0.50 * lowerCord.thickness;
        tube[85].Dx = tube[85].Dxeq = 0.25 * 4.0 * f + 0.75 * lowerCord.thickness;
        
        tube[78].Dy = tube[78].Dyeq = 11.0 * f;
        tube[79].Dy = tube[79].Dyeq = 7.0 * f;
        tube[80].Dy = tube[80].Dyeq = 4.0 * f;
        tube[81].Dy = tube[81].Dyeq = 2.0 * f;
        tube[82].Dy = tube[82].Dyeq = 1.0 * f;
        tube[83].Dy = tube[83].Dyeq = 0.75 * f;
        tube[84].Dy = tube[84].Dyeq = 0.50 * f;
        tube[85].Dy = tube[85].Dyeq = 0.25 * f;
        
        tube[78].Dz = tube[78].Dzeq = 16.0 * f;
        tube[79].Dz = tube[79].Dzeq = 16.0 * f;
        tube[80].Dz = tube[80].Dzeq = 16.0 * f;
        tube[81].Dz = tube[81].Dzeq = 16.0 * f;
        tube[82].Dz = tube[82].Dzeq = 16.0 * f;
        tube[83].Dz = tube[83].Dzeq = 0.75 * 16.0 * f + 0.25 * cord.length;
        tube[84].Dz = tube[84].Dzeq = 0.50 * 16.0 * f + 0.50 * cord.length;
        tube[85].Dz = tube[85].Dzeq = 0.25 * 16.0 * f + 0.75 * cord.length;
        
        tube[78].k1 = 160.0;
        tube[79].k1 = 160.0;
        tube[80].k1 = 160.0;
        tube[81].k1 = 160.0;
        tube[82].k1 = 160.0;
        tube[83].k1 = 0.75 * 160.0 * f + 0.25 * lowerCord.k1;
        tube[84].k1 = 0.50 * 160.0 * f + 0.50 * lowerCord.k1;
        tube[85].k1 = 0.25 * 160.0 * f + 0.75 * lowerCord.k1;
        
        tube[78].Brel = 0.7;
        tube[79].Brel = 0.6;
        tube[80].Brel = 0.5;
        tube[81].Brel = 0.4;
        tube[82].Brel = 0.3;
        tube[83].Brel = 0.2;
        tube[84].Brel = 0.2;
        tube[85].Brel = 0.2;
        
        for (itube = 78; itube <= 85; itube ++) {
            Delta_Tube t = &(tube[itube]);
            t -> mass = t -> Dx * t -> Dz / (30.0 * f);
            t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
            t -> k1left1 = t -> k1right1 = 1.0;
        }
        tube[78].k1left1 = 0.0;
        tube[35].k1left1 = 1.0;   // the essence: couple spring with lower vocal cords
    }
    
     // Fill in the values of the glottal shunt only if we want to model it.
    if (shunt.Dx != 0.0) {
        for (itube = 86; itube <= 88; itube ++) {
            Delta_Tube t = &(tube[itube]);
            t -> Dx = t -> Dxeq = shunt.Dx;
            t -> Dy = t -> Dyeq = shunt.Dy;
            t -> Dz = t -> Dzeq = shunt.Dz;
            t -> mass = 3.0 * upperCord.mass;   // heavy...
            t -> k1 = 3.0 * upperCord.k1;   // ...and stiff...
            t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
            t -> Brel = 3.0;   // ...and inelastic, so that the walls will not vibrate
        }
    }
    
    // Vocal tract from neutral articulation.
    {
        // TODO: Add virtual art to vt class
        Articulation art_0 = {0.0}; // all values are defaulted to zero
        // TODO: Don't like having to call this and then the MeshSum () it is confusing.
        MeshUpper(art_0);
    }
    
    // Pharynx and mouth: tubes 37..63.
    for (itube = 37; itube <= 63; itube ++) {
        Delta_Tube t = &(tube[itube]);
        int i = itube - 36;
        // TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
        //       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
        t -> Dx = t -> Dxeq = MeshSumX(i);
        t -> Dyeq = MeshSumY(i);
        if (closed [i]) t -> Dyeq = - t -> Dyeq;
        t -> Dy = t -> Dyeq;
        t -> Dz = t -> Dzeq = 0.015;
        t -> mass = 0.006;
        t -> k1 = 30.0;
        t -> k3 = 0.0;
        t -> Brel = 1.0;
    }
    
    // For tongue-tip vibration [r]:  tube [59]. Brel = 0.1; tube [59]. k1 = 3;
    
    // Nose: tubes 64..77.
    
    for (itube = 64; itube <= 77; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> Dx = t -> Dxeq = nose.Dx;
        t -> Dy = t -> Dyeq = nose.weq [itube - 64]; // Zero indexing nose array
        t -> Dz = t -> Dzeq = nose.Dz;
        t -> mass = 0.006;
        t -> k1 = 100.0;
        t -> k3 = 0.0;
        t -> Brel = 1.0;
    }
    tube[64].Dy = tube[64].Dyeq = 0.0;   // override: nasopharyngeal port closed
    
    // The default structure: every tube is connected on the left to the previous tube (index one lower).
    // This corresponds to a two-mass model of the vocal cords without shunt.
    for (itube = SMOOTH_LUNGS ? FIRST_TUBE : 0; itube < numberOfTubes; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> s1 = 5e6 * t -> Dx * t -> Dz;
        t -> s3 = t -> s1 / (0.9e-3 * 0.9e-3);
        t -> dy = 1e-5;
        t -> left1 = &(tube[itube-1]);   // connect to the previous tube on the left
        // TODO: This is overrunning the buffer here, but it gets turned into a null pointer below. Is this a problem?
        /*
        if (itube==numberOfTubes-1)
         t -> right1 = nullptr;
        else
         t -> right1 = &(tube[itube+1]);   // connect to the next tube on the right
        */
        t -> right1 = &(tube[itube+1]);   // connect to the next tube on the right
        
    }
    // **** Connections: boundaries and interfaces. ***** //
    
    // The leftmost boundary: the diaphragm (tube 1). Disconnect on the left.
    tube[SMOOTH_LUNGS ? FIRST_TUBE : 0]. left1 = nullptr;   // closed at diaphragm
    
    // Optional one-mass model of the vocal cords. Short-circuit over tube 37 (upper glottis).
    if (cord.numberOfMasses == 1) {
        
        // Connect the right side of tube 35 to the left side of tube 37.
        tube[35]. right1 = &(tube[37]);
        tube[37]. left1 = &(tube[35]);
        
        // Disconnect tube 36 on both sides.
        tube[36].left1 = tube[36].right1 = nullptr;
    }
    
    // Optionally couple vocal cords with conus elasticus.
    // Replace tubes 31..34 (upper trachea) by tubes 78..85 (conus elasticus).
    if (cord.numberOfMasses == 10) {
        
        // Connect the right side of tube 30 to the left side of tube 78.
        tube[30].right1 = &(tube[78]);
        tube[78].left1 = &(tube[30]);
        
        // Connect the right side of tube 85 to the left side of tube 35.
        tube[85].right1 = &(tube[35]);
        tube[35].left1 = &(tube[85]);
        
        // Disconnect tubes 31..34 on both sides.
        tube[31].left1 = tube[31].right1 = nullptr;
        tube[32].left1 = tube[32].right1 = nullptr;
        tube[33].left1 = tube[33].right1 = nullptr;
        tube[34].left1 = tube[34].right1 = nullptr;
    } else {
        
        // Disconnect tubes 78..85 on both sides.
        for (itube = 78; itube <= 85; itube ++)
            tube[itube].left1 = tube[itube].right1 = nullptr;
    }
    
    // Optionally add a shunt parallel to the glottis.
    // Create a side branch from tube 33/34 (or 84/85) to tube 37/38 with tubes 86..88.
    if (shunt.Dx != 0.0) {
        int topOfTrachea = ( cord.numberOfMasses == 10 ? 85 : 34 );
        
        // Create a three-way interface below the shunt.
        // Connect lowest shunt tube (87) with top of trachea (33/34 or 84/85).
        tube[topOfTrachea - 1].right2 = &(tube[86]);   // trachea to shunt
        tube[86].left1 = &(tube[topOfTrachea - 1]);   // shunt to trachea
        tube[86].Dxeq = tube[topOfTrachea - 1].Dxeq = tube[topOfTrachea].Dxeq;   // equal length
        tube[86].Dx = tube[topOfTrachea - 1].Dx = tube[topOfTrachea].Dx;
        
        // Create a three-way interface above the shunt.
        // Connect highest shunt tube (88) with bottom of pharynx (37/38).
        tube[88].right1 = &(tube[38]);   // shunt to pharynx
        tube[38].left2 = &(tube[88]);   // pharynx to shunt
        tube[88].Dxeq = tube[38].Dxeq = tube[37].Dxeq;   // all three of equal length
        tube[88].Dx = tube[38].Dx = tube[37].Dx;
    } else {
        
        // Disconnect tubes 86..88 on both sides.
        for (itube = 86; itube <= 88; itube ++)
            tube[itube].left1 = tube[itube].right1 = nullptr;
    }
    
    // Create a three-way interface at the nasopharyngeal port.
    // Connect tubes 49 (pharynx), 50 (mouth), and 64 (nose).
    tube[49].right2 = &(tube[64]);   // pharynx to nose
    tube[64].left1 = &(tube[49]);   // nose to pharynx
    tube[64].Dxeq = tube[50].Dxeq = tube[49].Dxeq;   // all three must be of equal length
    tube[64].Dx = tube[50].Dx = tube[49].Dx;
    
    // The rightmost boundaries: the lips (tube 63) and the nostrils (tube 77).
    // Disconnect on the right.
    tube[63]. right1 = nullptr;   // radiation at the lips
    tube[77]. right1 = nullptr;   // radiation at the nostrils
    
    for (itube = 0; itube < numberOfTubes; itube ++) {
        Delta_Tube t = &(tube[itube]);
        assert(! t->left1 || t->left1->right1 == t || t->left1->right2 == t);
        assert(! t->left2 || t->left2->right1 == t);
        assert(! t->right1 || t->right1->left1 == t || t->right1->left2 == t);
        assert(! t->right2 || t->right2->left1 == t);
    }
    Dt = 1.0 / fsamp / oversamp,
    rho0 = 1.14,
    c = 353.0,
    onebyc2 = 1.0 / (c * c),
    rho0c2 = rho0 * c * c,
    halfDt = 0.5 * Dt,
    twoDt = 2.0 * Dt,
    halfc2Dt = 0.5 * c * c * Dt,
    twoc2Dt = 2.0 * c * c * Dt,
    onebytworho0 = 1.0 / (2.0 * rho0),
    Dtbytworho0 = Dt / (2.0 * rho0),
    rrad = 1.0 - c * Dt / 0.02,   // radiation resistance, 5.135
    onebygrad = 1.0 / (1.0 + c * Dt / 0.02);   // radiation conductance, 5.135
    tension = 0;
    #if NO_RADIATION_DAMPING
        rrad = 0;
        onebygrad = 0;
    #endif
}

void Speaker::UpdateTube()
{
	double f = relativeSize * 1e-3;
	int itube;

	// Lungs.

	for (itube = 6; itube <= 17; itube ++)
		tube[itube]. Dyeq = 120 * f * (1 + art [kArt_muscle_LUNGS]);

	// Glottis.

	{
		Delta_Tube t = &(tube[35]);
		t -> Dyeq = f * (5 - 10 * art [kArt_muscle_INTERARYTENOID]
		      + 3 * art [kArt_muscle_POSTERIOR_CRICOARYTENOID]
		      - 3 * art [kArt_muscle_LATERAL_CRICOARYTENOID]);   // 4.38
		t -> k1 = lowerCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
		t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
	}
	if (cord.numberOfMasses >= 2) {
		Delta_Tube t = &(tube[36]);
		t -> Dyeq = tube[35]. Dyeq;
		t -> k1 = upperCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
		t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
	}
	if (cord.numberOfMasses >= 10) {
		tube[83]. Dyeq = 0.75 * 1 * f + 0.25 * tube[35]. Dyeq;
		tube[84]. Dyeq = 0.50 * 1 * f + 0.50 * tube[35]. Dyeq;
		tube[85]. Dyeq = 0.25 * 1 * f + 0.75 * tube[35]. Dyeq;
		tube[83]. k1 = 0.75 * 160 + 0.25 * tube[35]. k1;
		tube[84]. k1 = 0.50 * 160 + 0.50 * tube[35]. k1;
		tube[85]. k1 = 0.25 * 160 + 0.75 * tube[35]. k1;
		for (itube = 83; itube <= 85; itube ++)
			tube[itube]. k3 = tube[itube]. k1 *
				(20 / tube[itube]. Dz) * (20 / tube[itube]. Dz);
	}

	// Vocal tract.

    // TODO: Don't like having to call this and then the MeshSum () it is confusing.
    MeshUpper(art);
	for (itube = 37; itube <= 63; itube ++) {
		Delta_Tube t = &(tube[itube]);
        // TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
        //       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
		int i = itube - 36;
		t -> Dxeq = MeshSumX(i);
		t -> Dyeq = MeshSumY(i);
		if (closed [i]) t -> Dyeq = - t -> Dyeq;
	}
	tube[64]. Dxeq = tube[50]. Dxeq = tube[49]. Dxeq;
	// Voor [r]:  thy tube [59]. Brel = 0.1; thy tube [59]. k1 = 3;

	// Nasopharyngeal port.

	tube[64]. Dyeq = f * (18 - 25 * art [kArt_muscle_LEVATOR_PALATINI]);   // 4.40

	for (itube = 0; itube < numberOfTubes; itube ++) {
        Delta_Tube t = &(tube[itube]);
		t -> s1 = 5e6 * t -> Dxeq * t -> Dzeq;
		t -> s3 = t -> s1 / (0.9e-3 * 0.9e-3);
	}
}

void Speaker::InitSim(double totalTime, Articulation initialArt)
{
	try {
        // TODO: Make Articulation a class and use either a copy funciton or overload =
        memcpy(art, initialArt, sizeof(Articulation));
        if(!result->IsInitialized()) {
            result->Initialize(1, totalTime, fsamp);
        }
        else {
            result->ResetArray(totalTime);
        }
		numberOfSamples = result -> numberOfSamples;
        sample = 0;

        UpdateTube();

		double totalVolume = 0.0;
		for (int m = 0; m < numberOfTubes; m ++) {
			Delta_Tube t = &(tube[m]);
			if (! t -> left1 && ! t -> right1) continue;
			t->Dx = t->Dxeq; t->dDxdt = 0.0;   // 5.113 (numbers refer to equations in Boersma (1998)
			t->Dy = t->Dyeq; t->dDydt = 0.0;   // 5.113
			t->Dz = t->Dzeq;   // 5.113
			t->A = t->Dz * ( t->Dy >= t->dy ? t->Dy + DYMIN :
				t->Dy <= - t->dy ? DYMIN :
				(t->dy + t->Dy) * (t->dy + t->Dy) / (4.0 * t->dy) + DYMIN );   // 4.4, 4.5
			#if EQUAL_TUBE_WIDTHS
				t->A = 0.0001;
			#endif
			t->Jleft = t->Jright = 0.0;   // 5.113
			t->Qleft = t->Qright = rho0c2;   // 5.113
			t->pleft = t->pright = 0.0;   // 5.114
			t->Kleft = t->Kright = 0.0;   // 5.114
            t->Pturbright = t->Pturbrightnew = 0.0;
			t->V = t->A * t->Dx;   // 5.114
            #if MASS_LEAPFROG
                t->ehalfold;
            #endif
			totalVolume += t->V;
		}
		printf("Starting volume: %f liters.\n", totalVolume * 1000);
        // Set up log counters and write headers
        if(log_data == true) {
            InitDataLogger();
        }

	} catch (int e) {
        std::cout << "Articulatory synthesizer not initialized.\n";
	}
}

void Speaker::IterateSim() 
{

    UpdateTube();
    // Oversample to simulate dynamics of the vocal tract walls
    for (int n = 1; n <= oversamp; n ++)
    {
        //Loop along each tube segment 
        for (int m = 0; m < numberOfTubes; m ++)
        {
            Delta_Tube t = &(tube[m]);
            if (! t -> left1 && ! t -> right1) continue;

            // New geometry.
            #if CONSTANT_TUBE_LENGTHS
                t->Dxnew = t->Dx;
            #else
                t->dDxdtnew = (t->dDxdt + Dt * 10000.0 * (t->Dxeq - t->Dx)) /
                    (1.0 + 200.0 * Dt);   // critical damping, 10 ms
                t->Dxnew = t->Dx + t->dDxdtnew * Dt;
            #endif

            // 3-way: equal lengths. 
            // This requires left tubes to be processed before right tubes. 
            if (t->left1 && t->left1->right2) t->Dxnew = t->left1->Dxnew;

            t->Dz = t->Dzeq;   // immediate... 
            t->eleft = (t->Qleft - t->Kleft) * t->V;   // 5.115
            t->eright = (t->Qright - t->Kright) * t->V;   // 5.115
            t->e = 0.5 * (t->eleft + t->eright);   // 5.116
            t->p = 0.5 * (t->pleft + t->pright);   // 5.116
            t->DeltaP = t->e / t->V - rho0c2;   // 5.117
            t->v = t->p / (rho0 + onebyc2 * t->DeltaP);   // 5.118

            { 
                double dDy = t->Dyeq - t->Dy;
                double cubic = t->k3 * dDy * dDy;
                Delta_Tube l1 = t->left1, l2 = t->left2, r1 = t->right1, r2 = t->right2;
                tension = dDy * (t->k1 + cubic);
                t->B = 2.0 * t->Brel * sqrt (t->mass * (t->k1 + 3.0 * cubic));
                if (t->k1left1 != 0.0 && l1)
                    tension += t->k1left1 * t->k1 * (dDy - (l1->Dyeq - l1->Dy));
                if (t->k1left2 != 0.0 && l2)
                    tension += t->k1left2 * t->k1 * (dDy - (l2->Dyeq - l2->Dy));
                if (t->k1right1 != 0.0 && r1)
                    tension += t->k1right1 * t->k1 * (dDy - (r1->Dyeq - r1->Dy));
                if (t->k1right2 != 0.0 && r2)
                    tension += t->k1right2 * t->k1 * (dDy - (r2->Dyeq - r2->Dy));
            }

            if (t->Dy < t->dy) {
                if (t->Dy >= - t->dy) {
                    double dDy = t->dy - t->Dy, dDy2 = dDy * dDy;
                    tension += dDy2 / (4.0 * t->dy) * (t->s1 + 0.5 * t->s3 * dDy2);
                    t->B += 2.0 * dDy / (2.0 * t->dy) *
                        sqrt (t->mass * (t->s1 + t->s3 * dDy2));
                } else {
                    tension -= t->Dy * (t->s1 + t->s3 * (t->Dy * t->Dy + t->dy * t->dy));
                    t->B += 2.0 * sqrt (t->mass * (t->s1 + t->s3 * (3.0 * t->Dy * t->Dy + t->dy * t->dy)));
                }
            }

            t->dDydtnew = (t->dDydt + Dt / t->mass * (tension + 2.0 * t->DeltaP * t->Dz * t->Dx)) / (1.0 + t->B * Dt / t->mass);   // 5.119
            t->Dynew = t->Dy + t->dDydtnew * Dt;   // 5.119
            #if NO_MOVING_WALLS
                t->Dynew = t->Dy;
            #endif
            t->Anew = t->Dz * ( t->Dynew >= t->dy ? t->Dynew + DYMIN :
                t->Dynew <= - t->dy ? DYMIN :
                (t->dy + t->Dynew) * (t->dy + t->Dynew) / (4.0 * t->dy) + DYMIN );   // 4.4, 4.5
            #if EQUAL_TUBE_WIDTHS
                t->Anew = 0.0001;
            #endif
            t->Ahalf = 0.5 * (t->A + t->Anew);   // 5.120
            t->Dxhalf = 0.5 * (t->Dxnew + t->Dx);   // 5.121
            t->Vnew = t->Anew * t->Dxnew;   // 5.128

            {
                double oneByDyav = t->Dz / t->A;
                //t->R = 12.0 * 1.86e-5 * t->parallel * t->parallel * oneByDyav * oneByDyav;
                if (t->Dy < 0.0)
                    t->R = 12.0 * 1.86e-5 / (DYMIN * DYMIN + t->dy * t->dy);
                else
                    t->R = 12.0 * 1.86e-5 * t->parallel * t->parallel /
                        ((t->Dy + DYMIN) * (t->Dy + DYMIN) + t->dy * t->dy);
                t->R += 0.3 * t->parallel * oneByDyav;   // 5.23 
            }

            t->r = (1.0 + t->R * Dt / rho0) * t->Dxhalf / t->Anew;   // 5.122
            t->ehalf = t->e + halfc2Dt * (t->Jleft - t->Jright);   // 5.123
            t->phalf = (t->p + halfDt * (t->Qleft - t->Qright) / t->Dx) / (1.0 + Dtbytworho0 * t->R);   // 5.123
            #if MASS_LEAPFROG
                t->ehalf = t->ehalfold + 2.0 * halfc2Dt * (t->Jleft - t->Jright);
            #endif
            t->Jhalf = t->phalf * t->Ahalf;   // 5.124
            t->Qhalf = t->ehalf / (t->Ahalf * t->Dxhalf) + onebytworho0 * t->phalf * t->phalf;   // 5.124
            #if NO_BERNOULLI_EFFECT
                t->Qhalf = t->ehalf / (t->Ahalf * t->Dxhalf);
            #endif
        }// end Tube segment loop

        // Loop tube segments again? Combining the loops makes it crap it's pants...
        for (int m = 0; m < numberOfTubes; m ++) {   // compute Jleftnew and Qleftnew
            // TODO: This is some confusing use of the , operator. It saves space, but makes it hard to read.
            Delta_Tube l = &(tube[m]), r1 = l -> right1, r2 = l -> right2, r = r1;
            Delta_Tube l1 = l, l2 = r ? r -> left2 : nullptr;

            if (! l->left1) {   // closed boundary at the left side (diaphragm)?
                if (! r) continue;   // tube not connected at all
                l->Jleftnew = 0;   // 5.132
                l->Qleftnew = (l->eleft - twoc2Dt * l->Jhalf) / l->Vnew;   // 5.132
            }
            else   // left boundary open to another tube will be handled...
                (void) 0;   // ...together with the right boundary of the tube to the left

            if (! r) {   // open boundary at the right side (lips, nostrils)?
                l->prightnew = ((l->Dxhalf / Dt + c * onebygrad) * l->pright +
                     2.0 * ((l->Qhalf - rho0c2) - (l->Qright - rho0c2) * onebygrad)) /
                    (l->r * l->Anew / Dt + c * onebygrad);   // 5.136
                l->Jrightnew = l->prightnew * l->Anew;   // 5.136
                l->Qrightnew = (rrad * (l->Qright - rho0c2) +
                    c * (l->prightnew - l->pright)) * onebygrad + rho0c2;   // 5.136
            } else if (! l2 && ! r2) {   // two-way boundary
                if (l->v > CRITICAL_VELOCITY && l->A < r->A) {
                    l->Pturbrightnew = -0.5 * rho0 * (l->v - CRITICAL_VELOCITY) *
                        (1.0 - l->A / r->A) * (1.0 - l->A / r->A) * l->v;
                    if (l->Pturbrightnew != 0.0)
                        l->Pturbrightnew *= distribution (generator); // * l->A; 
                }
                if (r->v < - CRITICAL_VELOCITY && r->A < l->A) {
                    l->Pturbrightnew = 0.5 * rho0 * (r->v + CRITICAL_VELOCITY) * (1.0 - r->A / l->A) * (1.0 - r->A / l->A) * r->v;
                    if (l->Pturbrightnew != 0.0)
                        l->Pturbrightnew *= distribution (generator); // * r->A ;
                }
                
                #if NO_TURBULENCE
                    l->Pturbrightnew = 0.0;
                #endif

                l->Jrightnew = r->Jleftnew = (l->Dxhalf * l->pright + r->Dxhalf * r->pleft + twoDt * (l->Qhalf - r->Qhalf + l->Pturbright)) / (l->r + r->r);   // 5.127

                #if B91
                    l->Jrightnew = r->Jleftnew = (l->pright + r->pleft + 2.0 * twoDt * (l->Qhalf - r->Qhalf + l->Pturbright) / (l->Dxhalf + r->Dxhalf)) / (l->r / l->Dxhalf + r->r / r->Dxhalf);
                #endif

                l->prightnew = l->Jrightnew / l->Anew;   // 5.128
                r->pleftnew = r->Jleftnew / r->Anew;   // 5.128
                l->Krightnew = onebytworho0 * l->prightnew * l->prightnew;   // 5.128
                r->Kleftnew = onebytworho0 * r->pleftnew * r->pleftnew;   // 5.128

                #if NO_BERNOULLI_EFFECT
                    l->Krightnew = r->Kleftnew = 0.0;
                #endif

                l->Qrightnew = (l->eright + r->eleft + twoc2Dt * (l->Jhalf - r->Jhalf) + l->Krightnew * l->Vnew + (r->Kleftnew - l->Pturbrightnew) * r->Vnew) / (l->Vnew + r->Vnew);   // 5.131
                r->Qleftnew = l->Qrightnew + l->Pturbrightnew;   // 5.131

            } else if (r2) {   // two adjacent tubes at the right side (velic)
                r1->Jleftnew = (r1->Jleft * r1->Dxhalf * (1.0 / (l->A + r2->A) + 1.0 / r1->A) +
                                twoDt * ((l->Ahalf * l->Qhalf + r2->Ahalf * r2->Qhalf ) / (l->Ahalf  + r2->Ahalf) - r1->Qhalf)) /
                                (1.0 / (1.0 / l->r + 1.0 / r2->r) + r1->r);   // 5.138
                r2->Jleftnew = (r2->Jleft * r2->Dxhalf * (1.0 / (l->A + r1->A) + 1.0 / r2->A) +
                                twoDt * ((l->Ahalf * l->Qhalf + r1->Ahalf * r1->Qhalf ) / (l->Ahalf  + r1->Ahalf) - r2->Qhalf)) /
                                (1.0 / (1.0 / l->r + 1.0 / r1->r) + r2->r);   // 5.138
                l->Jrightnew = r1->Jleftnew + r2->Jleftnew;   // 5.139
                l->prightnew = l->Jrightnew / l->Anew;   // 5.128
                r1->pleftnew = r1->Jleftnew / r1->Anew;   // 5.128
                r2->pleftnew = r2->Jleftnew / r2->Anew;   // 5.128
                l->Krightnew = onebytworho0 * l->prightnew * l->prightnew;   // 5.128
                r1->Kleftnew = onebytworho0 * r1->pleftnew * r1->pleftnew;   // 5.128
                r2->Kleftnew = onebytworho0 * r2->pleftnew * r2->pleftnew;   // 5.128

                #if NO_BERNOULLI_EFFECT
                    l->Krightnew = r1->Kleftnew = r2->Kleftnew = 0;
                #endif

                l->Qrightnew = r1->Qleftnew = r2->Qleftnew =
                    (l->eright + r1->eleft + r2->eleft + twoc2Dt * (l->Jhalf - r1->Jhalf - r2->Jhalf) +
                     l->Krightnew * l->Vnew + r1->Kleftnew * r1->Vnew + r2->Kleftnew * r2->Vnew) /
                    (l->Vnew + r1->Vnew + r2->Vnew);   // 5.137
            } else {
                assert (l2 != nullptr);
                l1->Jrightnew =
                    (l1->Jright * l1->Dxhalf * (1.0 / (r->A + l2->A) + 1.0 / l1->A) -
                     twoDt * ((r->Ahalf * r->Qhalf + l2->Ahalf * l2->Qhalf ) / (r->Ahalf  + l2->Ahalf) - l1->Qhalf)) /
                    (1.0 / (1.0 / r->r + 1.0 / l2->r) + l1->r);   // 5.138
                l2->Jrightnew =
                    (l2->Jright * l2->Dxhalf * (1.0 / (r->A + l1->A) + 1.0 / l2->A) -
                     twoDt * ((r->Ahalf * r->Qhalf + l1->Ahalf  * l1->Qhalf ) / (r->Ahalf  + l1->Ahalf) - l2->Qhalf)) /
                    (1.0 / (1.0 / r->r + 1.0 / l1->r) + l2->r);   // 5.138
                r->Jleftnew = l1->Jrightnew + l2->Jrightnew;   // 5.139
                r->pleftnew = r->Jleftnew / r->Anew;   // 5.128
                l1->prightnew = l1->Jrightnew / l1->Anew;   // 5.128
                l2->prightnew = l2->Jrightnew / l2->Anew;   // 5.128
                r->Kleftnew = onebytworho0 * r->pleftnew * r->pleftnew;   // 5.128
                l1->Krightnew = onebytworho0 * l1->prightnew * l1->prightnew;   // 5.128
                l2->Krightnew = onebytworho0 * l2->prightnew * l2->prightnew;   // 5.128
                #if NO_BERNOULLI_EFFECT
                    r->Kleftnew = l1->Krightnew = l2->Krightnew = 0.0;
                #endif
                r->Qleftnew = l1->Qrightnew = l2->Qrightnew =
                    (r->eleft + l1->eright + l2->eright + twoc2Dt * (l1->Jhalf + l2->Jhalf - r->Jhalf) +
                     r->Kleftnew * r->Vnew + l1->Krightnew * l1->Vnew + l2->Krightnew * l2->Vnew) /
                    (r->Vnew + l1->Vnew + l2->Vnew);   // 5.137
            }
            if (isnan(tube[m].Jrightnew)||isinf(tube[m].Jrightnew)||isnan(tube[m].Jleftnew)||isinf(tube[m].Jleftnew)||isnan(tube[m].Kleftnew)||isinf(tube[m].Kleftnew)||isnan(tube[m].Qleftnew)||isinf(tube[m].Qleftnew)||isnan(tube[m].Anew)||isinf(tube[m].Anew)) {
                int temp = 1;
            }
        } // end second tube loop 

        last_snd = ComputeSound();
        // Save Sound at middle sample
        if (n == ((long)oversamp+ 1) / 2) {
            result->z[sample] = last_snd;
        }

        // Increment tube parameters for next iteration
        for (int m = 0; m < numberOfTubes; m ++) {
            Delta_Tube t = &(tube[m]);
            t->Jleft = t->Jleftnew;
            t->Jright = t->Jrightnew;
            t->Qleft = t->Qleftnew;
            t->Qright = t->Qrightnew;
            t->Dy = t->Dynew;
            t->dDydt = t->dDydtnew;
            t->A = t->Anew;
            t->Dx = t->Dxnew;
            t->dDxdt = t->dDxdtnew;
            t->eleft = t->eleftnew;
            t->eright = t->erightnew;
            #if MASS_LEAPFROG
                t->ehalfold = t->ehalf;
            #endif
            t->pleft = t->pleftnew;
            t->pright = t->prightnew;
            t->Kleft = t->Kleftnew;
            t->Kright = t->Krightnew;
            t->V = t->Vnew;
            t->Pturbright = t->Pturbrightnew;
        }

    } // End oversample loop
    // Outupt some data to log file
    if (log_data) {
        Log();
    }
    ++sample;
    if (!NotDone()) {
        printf("Ending volume: %f liters.\n", getVolume() * 1000);
    }
}

int Speaker::Speak() 
{
    return result->play();
}

double Speaker::ComputeSound()
{
    double out = 0.0;
    for (int m = 0; m < numberOfTubes; m ++)
    {
        Delta_Tube t = &(tube[m]);
        out += rho0 * t->Dx * t->Dz * t->dDydt * Dt * 1000.0;   // radiation of wall movement, 5.140
        // Don't perform difference if this is the first logsample, because these values could be held over from the last run of the sim
        if (! t->right1 && logSample!=0)
            out += t->Jrightnew - t->Jright;   // radiation of open tube end
    }
    out /= 4.0 * M_PI * 0.4 * Dt;
    return out;
}

double Speaker::getVolume() {
    double totalVolume = 0.0;
    for (int m = 0; m < numberOfTubes; m ++)
        totalVolume += tube [m].V;
    return totalVolume;
}

void Speaker::getAreaFcn(AreaFcn AreaFcn_) {
    for(int ind=0; ind<numberOfTubes; ind++)
    {
        AreaFcn_[ind] = tube[ind].A;
    }
}

void Speaker::Log()
{
    if(logCounter +1 == log_period)
    {
        for(int ind=0; ind<numberOfTubes; ind++)
        {
            //For logging area
            *log_stream << tube[ind].A;
            *log_stream << "\t";
            
            /* For logging Dx, Dy, Dz
            *log_stream << tube[ind].Dxnew;
            *log_stream << "\t";
            *log_stream << tube[ind].Dynew;
            *log_stream << "\t";
            *log_stream << tube[ind].Dz;
            *log_stream << "\t";
             */
        }
        for(int ind=0; ind<kArt_muscle_MAX; ind++)
        {
            *log_stream << art[ind];
            *log_stream << "\t";
        }
        *log_stream << last_snd;
        *log_stream << "\n";
        
        ++logSample;
        if (logSample==numberOfLogSamples)
        {
            // TODO: Think about if we need to do anything here or not
            log_stream->flush();
            log_data = false;
        }
        logCounter = 0;
        return;
    }
    ++logCounter;
}

int Speaker::ConfigDataLogger(std::string filepath, int _log_period)
{
    log_data = true;
    if( log_stream == nullptr) {
        log_stream = new std::ofstream(filepath);
        // Ensure that all of the digits are written out. I think this ensures we have the correct precision.
        // Using 32 digits of precision to get the best accuracy I can without using binary or hex values in the log files
        //log_stream->precision(std::numeric_limits<double>::digits10); // Is 15 digits
        // TODO: Use hex or binary log files
        log_stream->precision(32);
        log_stream->setf(ios::scientific);
    }
    else {
        log_stream->close();
        log_stream->clear();
        log_stream->open(filepath);
    }
    log_period = _log_period;
    if(!log_stream)
    {
        exit(1);
    }
    return 0;
}

void Speaker::InitDataLogger()
{
    // Take an sample at time 0 and then every log_period steps after.
    // This means that we will not take a sample at the end of the sequence unless it is divisible by the log_period
    numberOfLogSamples = floor(result->duration*(fsamp/log_period))+1;

    *log_stream << "Sampling Frequency :\n";
    *log_stream << fsamp/log_period;
    *log_stream << "\n";
    *log_stream << "Number of Samples :\n" ;
    *log_stream << numberOfLogSamples;
    *log_stream << "\n\n";
    for(int ind=0; ind<numberOfTubes; ind++)
    {
        // Logging VT Area
        *log_stream << "A";
        *log_stream << ind;
        *log_stream << "\t";
        /* For logging Dx Dy Dz
        *log_stream << ind;
        *log_stream << "X\t";
        *log_stream << ind;
        *log_stream << "Y\t";
        *log_stream << ind;
        *log_stream << "Z\t";
         */
    }
    for(int ind=0; ind<kArt_muscle_MAX; ind++)
    {
        *log_stream << "Art ";
        *log_stream << ind;
        *log_stream << "\t";
    }
    *log_stream << "Sound\n";
    
    // Setup logger to take initial sample
    logCounter = log_period-1;
    logSample = 0;
    // Log initial sample of data
    Log();
}


int Speaker::SaveSound(std::string filepath)
{
    return result->save(filepath);
}
