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
#include <assert.h>
#define SMOOTH_LUNGS  true
#define FIRST_TUBE  6
#include "support_functions.h"


#include <iostream> //necessary?
#include <thread>

#include <algorithm>
#include <iostream>
#include <fstream>


/* constants for modifying acoustic simulation */
#define Dymin  0.00001
#define criticalVelocity  10.0

#define noiseFactor  0.1

#define MONITOR_SAMPLES  100

// While debugging, some of these can be 1; otherwise, they are all 0: 
#define EQUAL_TUBE_WIDTHS  0
#define CONSTANT_TUBE_LENGTHS  0 // was set to 1
#define NO_MOVING_WALLS  0
#define NO_TURBULENCE  0
#define NO_RADIATION_DAMPING  0
#define NO_BERNOULLI_EFFECT  0
#define MASS_LEAPFROG  0
#define B91  0

#define INTERMEDIATE_SOUNDS 0 // currently not supported...

// end acoustic simulation constants

using namespace std;

Speaker::Speaker(string kindOfSpeaker, int numberOfVocalCordMasses, double samplefreq, int oversamplefreq):
    fsamp(samplefreq),
    oversamp(oversamplefreq),
    distribution(1.0, noiseFactor)
{
    /* Preconditions:								
    *    1 <= numberOfVocalCordMasses <= 2;					
    * Failures:									
    *    Kind of speaker is not one of "Female", "Male", or "Child".	*/	

	/* Supralaryngeal dimensions are taken from P. Mermelstein (1973):		
	*    "Articulatory model for the study of speech production",		
	*    Journal of the Acoustical Society of America 53,1070 - 1082.	
	* That was a male speaker, so we need scaling for other speakers:		*/

	double scaling;
	if (kindOfSpeaker.compare("Male") == 0){ relativeSize = 1.1;}
    else if (kindOfSpeaker.compare("Child") == 0) { relativeSize = 0.7;}
	else { relativeSize = 1.0; }
	scaling = relativeSize;

	/* Laryngeal system. Data for male speaker from Ishizaka and Flanagan.	*/

	if (kindOfSpeaker.compare("Female")==0) {
		lowerCord.thickness = 1.4e-3;   // dx, in metres
		upperCord.thickness = 0.7e-3;
		cord.length = 10e-3;
		lowerCord.mass = 0.02e-3;   // kilograms
		upperCord.mass = 0.01e-3;
		lowerCord.k1 = 10;   // Newtons per metre
		upperCord.k1 = 4;
	} else if (kindOfSpeaker.compare("Male")==0) {
        lowerCord.thickness = 2.0e-3;   // dx, in metres
        upperCord.thickness = 1.0e-3;
        cord.length = 18e-3;
        lowerCord.mass = 0.1e-3;   // kilograms
        upperCord.mass = 0.05e-3;
        lowerCord.k1 = 12;   // Newtons per metre
        upperCord.k1 = 4;
    } else /* "Child" */ {
        lowerCord.thickness = 0.7e-3;   // dx, in metres
        upperCord.thickness = 0.3e-3;
        cord.length = 6e-3;
        lowerCord.mass = 0.003e-3;   // kilograms
        upperCord.mass = 0.002e-3;
        lowerCord.k1 = 6;   // Newtons per metre
        upperCord.k1 = 2;
    }
	cord.numberOfMasses = numberOfVocalCordMasses;
	if (numberOfVocalCordMasses == 1) {
		lowerCord.thickness += upperCord.thickness;
		lowerCord.mass += upperCord.mass;
		lowerCord.k1 += upperCord.k1;
	}
    
    shunt.Dx = 0;
    shunt.Dy = 0;
    shunt.Dz = 0;

	/* Supralaryngeal system. Data from Mermelstein. */

	velum.x = -0.031 * scaling;
	velum.y = 0.023 * scaling;
	velum.a = atan2 (velum.y, velum.x);
	palate.radius = sqrt (velum.x * velum.x + velum.y * velum.y);
	tip.length = 0.034 * scaling;
	neutralBodyDistance = 0.086 * scaling;
	alveoli.x = 0.024 * scaling;
	alveoli.y = 0.0302 * scaling;
	alveoli.a = atan2 (alveoli.y, alveoli.x);
	teethCavity.dx1 = -0.009 * scaling;
	teethCavity.dx2 = -0.004 * scaling;
	teethCavity.dy = -0.011 * scaling;
	lowerTeeth.a = -0.30;   // radians
	lowerTeeth.r = 0.113 * scaling;   // metres
	upperTeeth.x = 0.036 * scaling;
	upperTeeth.y = 0.026 * scaling;
	lowerLip.dx = 0.010 * scaling;
	lowerLip.dy = -0.004 * scaling;
	upperLip.dx = 0.010 * scaling;
	upperLip.dy = 0.004 * scaling;

	nose.Dx = 0.007 * scaling;
	nose.Dz = 0.014 * scaling;
	nose.weq [0] = 0.018 * scaling;
	nose.weq [1] = 0.016 * scaling;
	nose.weq [2] = 0.014 * scaling;
	nose.weq [3] = 0.020 * scaling;
	nose.weq [4] = 0.023 * scaling;
	nose.weq [5] = 0.020 * scaling;
	nose.weq [6] = 0.035 * scaling;
	nose.weq [7] = 0.035 * scaling;
	nose.weq [8] = 0.030 * scaling;
	nose.weq [9] = 0.022 * scaling;
	nose.weq [10] = 0.016 * scaling;
	nose.weq [11] = 0.010 * scaling;
	nose.weq [12] = 0.012 * scaling;
	nose.weq [13] = 0.013 * scaling;

    InitializeTube();
    result = new Sound();
}

//void Speaker_to_Delta (Speaker &me, Delta &thee) {
void Speaker::InitializeTube()
{
    // Map speaker parameters into delta tube
    // TODO: Determine wheter it is necessary to call this each time we run a new articulation with the same speaker
    Speaker_to_Delta(*this, delta);
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

//TODO: Add a reset function to reset the logCounter

//void Art_Speaker_intoDelta (Art &art, Speaker &speaker, Delta &delta)
// TODO: MOVE THIS FUNCTION INTO DELTA.CPP
void Speaker::UpdateTube()
{
    Speaker &speaker = *this;
    
	double f = speaker.relativeSize * 1e-3;
	double xe [30], ye [30], xi [30], yi [30], xmm [30], ymm [30], dx, dy;
	int closed [40];
	int itube;

	// Lungs.

	for (itube = 6; itube <= 17; itube ++)
		delta.tube[itube]. Dyeq = 120 * f * (1 + art [kArt_muscle_LUNGS]);

	// Glottis.

	{
		Delta_Tube t = &(delta.tube[35]);
		t -> Dyeq = f * (5 - 10 * art [kArt_muscle_INTERARYTENOID]
		      + 3 * art [kArt_muscle_POSTERIOR_CRICOARYTENOID]
		      - 3 * art [kArt_muscle_LATERAL_CRICOARYTENOID]);   // 4.38
		t -> k1 = speaker.lowerCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
		t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
	}
	if (speaker.cord.numberOfMasses >= 2) {
		Delta_Tube t = &(delta.tube[36]);
		t -> Dyeq = delta.tube[35]. Dyeq;
		t -> k1 = speaker.upperCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
		t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
	}
	if (speaker.cord.numberOfMasses >= 10) {
		delta.tube[83]. Dyeq = 0.75 * 1 * f + 0.25 * delta.tube[35]. Dyeq;
		delta.tube[84]. Dyeq = 0.50 * 1 * f + 0.50 * delta.tube[35]. Dyeq;
		delta.tube[85]. Dyeq = 0.25 * 1 * f + 0.75 * delta.tube[35]. Dyeq;
		delta.tube[83]. k1 = 0.75 * 160 + 0.25 * delta.tube[35]. k1;
		delta.tube[84]. k1 = 0.50 * 160 + 0.50 * delta.tube[35]. k1;
		delta.tube[85]. k1 = 0.25 * 160 + 0.75 * delta.tube[35]. k1;
		for (itube = 83; itube <= 85; itube ++)
			delta.tube[itube]. k3 = delta.tube[itube]. k1 *
				(20 / delta.tube[itube]. Dz) * (20 / delta.tube[itube]. Dz);
	}

	// Vocal tract.

	Art_Speaker_meshVocalTract (art, speaker, xi, yi, xe, ye, xmm, ymm, closed);
	for (itube = 37; itube <= 63; itube ++) {
		Delta_Tube t = &(delta.tube[itube]);
        // TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
        //       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
		int i = itube - 36;
		t -> Dxeq = sqrt (( dx = xmm [i] - xmm [i + 1], dx * dx ) + ( dy = ymm [i] - ymm [i + 1], dy * dy ));
		t -> Dyeq = sqrt (( dx = xe [i] - xi [i], dx * dx ) + ( dy = ye [i] - yi [i], dy * dy ));
		if (closed [i]) t -> Dyeq = - t -> Dyeq;
	}
	delta.tube[64]. Dxeq = delta.tube[50]. Dxeq = delta.tube[49]. Dxeq;
	// Voor [r]:  thy tube [59]. Brel = 0.1; thy tube [59]. k1 = 3;

	// Nasopharyngeal port.

	delta.tube[64]. Dyeq = f * (18 - 25 * art [kArt_muscle_LEVATOR_PALATINI]);   // 4.40

	for (itube = 0; itube < delta.numberOfTubes; itube ++) {
        Delta_Tube t = &(delta.tube[itube]);
		t -> s1 = 5e6 * t -> Dxeq * t -> Dzeq;
		t -> s3 = t -> s1 / (0.9e-3 * 0.9e-3);
	}
}


void Speaker::InitSim(double totalTime, std::string filepath, double log_freq)
{
	try {
        InitializeTube();
        if(!result->IsInitialized()) {
            result->Initialize(1, totalTime, fsamp);
        }
        else {
            result->ResetArray(totalTime);
        }
        // Test if the user wants to log data or not
        if (!filepath.empty()) {
            assert(log_freq>0);
            InitDataLogger( filepath, log_freq);
        }
		numberOfSamples = result -> numberOfSamples;
        sample = 0;

		/* TODO: Add in some sort of graphics.
        double minTract [1+78], maxTract [1+78];   // for drawing
         */

        UpdateTube();
		M = delta.numberOfTubes;

		/* TODO: Add in some sort of graphics.
        // Initialize drawing.
		for (int i = 1; i <= 78; i ++) {
			minTract [i] = 100.0;
			maxTract [i] = -100.0;
		} */

		totalVolume = 0.0;
		for (int m = 0; m < M; m ++) {
			Delta_Tube t = &(delta.tube[m]);
			if (! t -> left1 && ! t -> right1) continue;
			t->Dx = t->Dxeq; t->dDxdt = 0.0;   // 5.113 (numbers refer to equations in Boersma (1998)
			t->Dy = t->Dyeq; t->dDydt = 0.0;   // 5.113
			t->Dz = t->Dzeq;   // 5.113
			t->A = t->Dz * ( t->Dy >= t->dy ? t->Dy + Dymin :
				t->Dy <= - t->dy ? Dymin :
				(t->dy + t->Dy) * (t->dy + t->Dy) / (4.0 * t->dy) + Dymin );   // 4.4, 4.5
			#if EQUAL_TUBE_WIDTHS
				t->A = 0.0001;
			#endif
			t->Jleft = t->Jright = 0.0;   // 5.113
			t->Qleft = t->Qright = rho0c2;   // 5.113
			t->pleft = t->pright = 0.0;   // 5.114
			t->Kleft = t->Kright = 0.0;   // 5.114
			t->V = t->A * t->Dx;   // 5.114
			totalVolume += t->V;
		}
		//Melder_casual (U"Starting volume: ", totalVolume * 1000, U" litres.");

	} catch (int e) {
        std::cout << "Articulatory synthesizer not initialized.\n";
	}
}

void Speaker::IterateSim() 
{
			/* TODO: Add in some sort of graphics.
             if (sample % MONITOR_SAMPLES == 0 && monitor.graphics()) {   // because we can be in batch
				Graphics graphics = monitor.graphics();
				double area [1+78];
				for (int i = 1; i <= 78; i ++) {
					area [i] = delta -> tube [i]. A;
					if (area [i] < minTract [i]) minTract [i] = area [i];
					if (area [i] > maxTract [i]) maxTract [i] = area [i];
				}
				Graphics_beginMovieFrame (graphics, & Graphics_WHITE);

				Graphics_Viewport vp = Graphics_insetViewport (monitor.graphics(), 0.0, 0.5, 0.5, 1.0);
				Graphics_setWindow (graphics, 0.0, 1.0, 0.0, 0.05);
				Graphics_setColour (graphics, Graphics_RED);
				Graphics_function (graphics, minTract, 1, 35, 0.0, 0.9);
				Graphics_function (graphics, maxTract, 1, 35, 0.0, 0.9);
				Graphics_setColour (graphics, Graphics_BLACK);
				Graphics_function (graphics, area, 1, 35, 0.0, 0.9);
				Graphics_setLineType (graphics, Graphics_DOTTED);
				Graphics_line (graphics, 0.0, 0.0, 1.0, 0.0);
				Graphics_setLineType (graphics, Graphics_DRAWN);
				Graphics_resetViewport (graphics, vp);

				vp = Graphics_insetViewport (graphics, 0, 0.5, 0, 0.5);
				Graphics_setWindow (graphics, 0.0, 1.0, -0.000003, 0.00001);
				Graphics_setColour (graphics, Graphics_RED);
				Graphics_function (graphics, minTract, 36, 37, 0.2, 0.8);
				Graphics_function (graphics, maxTract, 36, 37, 0.2, 0.8);
				Graphics_setColour (graphics, Graphics_BLACK);
				Graphics_function (graphics, area, 36, 37, 0.2, 0.8);
				Graphics_setLineType (graphics, Graphics_DOTTED);
				Graphics_line (graphics, 0.0, 0.0, 1.0, 0.0);
				Graphics_setLineType (graphics, Graphics_DRAWN);
				Graphics_resetViewport (graphics, vp);

				vp = Graphics_insetViewport (graphics, 0.5, 1.0, 0.5, 1.0);
				Graphics_setWindow (graphics, 0.0, 1.0, 0.0, 0.001);
				Graphics_setColour (graphics, Graphics_RED);
				Graphics_function (graphics, minTract, 38, 64, 0.0, 1.0);
				Graphics_function (graphics, maxTract, 38, 64, 0.0, 1.0);
				Graphics_setColour (graphics, Graphics_BLACK);
				Graphics_function (graphics, area, 38, 64, 0.0, 1.0);
				Graphics_setLineType (graphics, Graphics_DOTTED);
				Graphics_line (graphics, 0.0, 0.0, 1.0, 0.0);
				Graphics_setLineType (graphics, Graphics_DRAWN);
				Graphics_resetViewport (graphics, vp);

				vp = Graphics_insetViewport (graphics, 0.5, 1.0, 0.0, 0.5);
				Graphics_setWindow (graphics, 0.0, 1.0, 0.001, 0.0);
				Graphics_setColour (graphics, Graphics_RED);
				Graphics_function (graphics, minTract, 65, 78, 0.5, 1.0);
				Graphics_function (graphics, maxTract, 65, 78, 0.5, 1.0);
				Graphics_setColour (graphics, Graphics_BLACK);
				Graphics_function (graphics, area, 65, 78, 0.5, 1.0);
				Graphics_setLineType (graphics, Graphics_DRAWN);
				Graphics_resetViewport (graphics, vp);

				Graphics_endMovieFrame (graphics, 0.0);
				Melder_monitor ((double) sample / numberOfSamples, U"Articulatory synthesis: ", Melder_half (time), U" seconds");
			} */

    UpdateTube();

    //
    // Oversample to simulate dynamics of the vocal tract walls
    //
    for (int n = 1; n <= oversamp; n ++) {

        //Loop along each tube segment 
        for (int m = 0; m < M; m ++) {

            //UpdateSegment(m); // only defined for the purpose of threading. Remove and uncomment section below to speed up
            // causes program to slow WAAAAAY down.
            //std::thread mythread(&Speaker::UpdateSegment, this, m);
            //mythread.detach();
            //
            Delta_Tube t = &(delta.tube[m]);
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
            t->Anew = t->Dz * ( t->Dynew >= t->dy ? t->Dynew + Dymin :
                t->Dynew <= - t->dy ? Dymin :
                (t->dy + t->Dynew) * (t->dy + t->Dynew) / (4.0 * t->dy) + Dymin );   // 4.4, 4.5
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
                    t->R = 12.0 * 1.86e-5 / (Dymin * Dymin + t->dy * t->dy);
                else
                    t->R = 12.0 * 1.86e-5 * t->parallel * t->parallel /
                        ((t->Dy + Dymin) * (t->Dy + Dymin) + t->dy * t->dy);
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
                //*/
        }// end Tube segment loop

        // Loop tube segments again? Combining the loops makes it crap it's pants...
        for (int m = 0; m < M; m ++) {   // compute Jleftnew and Qleftnew
            // TODO: This is some confusing use of the , operator. It saves space, but makes it hard to read.
            Delta_Tube l = &(delta.tube[m]), r1 = l -> right1, r2 = l -> right2, r = r1;
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
                if (l->v > criticalVelocity && l->A < r->A) {
                    l->Pturbrightnew = -0.5 * rho0 * (l->v - criticalVelocity) *
                        (1.0 - l->A / r->A) * (1.0 - l->A / r->A) * l->v;
                    if (l->Pturbrightnew != 0.0)
                        l->Pturbrightnew *= distribution (generator); // * l->A; 
                }
                if (r->v < - criticalVelocity && r->A < l->A) {
                    l->Pturbrightnew = 0.5 * rho0 * (r->v + criticalVelocity) * (1.0 - r->A / l->A) * (1.0 - r->A / l->A) * r->v;
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
        } // end second tube loop 

        // Save Sound at middle sample
        if (n == ((long)oversamp+ 1) / 2) {
            result->z[sample] = ComputeSound();
        }
        // Outupt some data to log file
        if (log_data) {
            Log();
        }

        // Increment tube parameters for next iteration
        for (int m = 0; m < M; m ++) {
            Delta_Tube t = &(delta.tube[m]);
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
    ++sample;
}
/* End of file Artword_Speaker_to_Sound.cpp */

int Speaker::Speak() 
{
    return result->play();
}

double Speaker::ComputeSound()
{
    double out = 0.0;
    for (int m = 0; m < M; m ++)
    {
        Delta_Tube t = &(delta.tube[m]);
        out += rho0 * t->Dx * t->Dz * t->dDydt * Dt * 1000.0;   // radiation of wall movement, 5.140
        if (! t->right1)
            out += t->Jrightnew - t->Jright;   // radiation of open tube end
    }
    out /= 4.0 * M_PI * 0.4 * Dt;
    return out;
}

void Speaker::Log()
{
    if(logCounter == numberOfOversampLogSamples)
    {
        for(int ind=0; ind<delta.numberOfTubes; ind++)
        {
            if(logSample+2==numberOfLogSamples && ind == 39)
            {double bannana = 1;}
            *log_stream << delta.tube[ind].Dxnew;
            *log_stream << "\t";
            *log_stream << delta.tube[ind].Dynew;
            *log_stream << "\t";
            *log_stream << delta.tube[ind].Dz;
            *log_stream << "\t";
        }
        for(int ind=0; ind<kArt_muscle_MAX; ind++)
        {
            *log_stream << art[ind];
            *log_stream << "\t";
        }
        *log_stream << ComputeSound();
        *log_stream << "\n";
        if (logSample+1==numberOfLogSamples)
        {
            // TODO: Think about if we need to do anything here or not
            //log_stream->close();
            log_data = false;
        }
        ++logSample;
        logCounter = 0;
    }
    ++logCounter;
}

int Speaker::InitDataLogger(std::string filepath, double log_freq)
{
    // !!! This should be called only after InitSim() is called
    log_data = true;
    if( log_stream == nullptr) {
        log_stream = new std::ofstream(filepath);
    }
    else {
        log_stream->close();
        log_stream->clear();
        log_stream->open(filepath);
    }
    logfreq = log_freq;
    numberOfOversampLogSamples = round((oversamp*fsamp)/logfreq);
    numberOfLogSamples = result->duration*logfreq+1; //TODO: This could be wrong
    logCounter = numberOfOversampLogSamples; // Setup logger to take first sample
    logSample = 0;
    if(!log_stream)
    {
        exit(1);
    }
    *log_stream << "Desired Sampling Frequency :";
    *log_stream << logfreq;
    *log_stream << "\n";
    *log_stream << "Actual Sampling Frequency :";
    *log_stream << logfreq;
    *log_stream << "\n";
    *log_stream << "Number of Samples :" ;
    *log_stream << numberOfLogSamples;
    *log_stream << "\n\n";
    for(int ind=0; ind<delta.numberOfTubes; ind++)
    {
        *log_stream << ind;
        *log_stream << "X\t";
        *log_stream << ind;
        *log_stream << "Y\t";
        *log_stream << ind;
        *log_stream << "Z\t";
    }
    for(int ind=0; ind<kArt_muscle_MAX; ind++)
    {
        *log_stream << "Art ";
        *log_stream << ind;
        *log_stream << "\t";
    }
    *log_stream << "Sound\n";
    return 0;
}


int Speaker::SaveSound(std::string filepath)
{
    return result->save(filepath);
}

/*void inline Speaker::UpdateSegment(int m) 
{
    Delta_Tube t = &(delta.tube[m]);
    if (! t -> left1 && ! t -> right1) return;

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
    t->Anew = t->Dz * ( t->Dynew >= t->dy ? t->Dynew + Dymin :
        t->Dynew <= - t->dy ? Dymin :
        (t->dy + t->Dynew) * (t->dy + t->Dynew) / (4.0 * t->dy) + Dymin );   // 4.4, 4.5
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
            t->R = 12.0 * 1.86e-5 / (Dymin * Dymin + t->dy * t->dy);
        else
            t->R = 12.0 * 1.86e-5 * t->parallel * t->parallel /
                ((t->Dy + Dymin) * (t->Dy + Dymin) + t->dy * t->dy);
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
}// end Tube segment loop */
