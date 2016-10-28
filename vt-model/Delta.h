#ifndef _Delta_h_
#define _Delta_h_
/* Delta.h
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
#include <vector>
#include <memory>
#include <assert.h>
#define MAX_NUMBER_OF_TUBES 89

struct structDelta_Tube{
    /* Structure: static. */
    structDelta_Tube* left1 = nullptr;   /* If null: closed at left edge. */
    structDelta_Tube* left2 = nullptr;   /* If not null: two merging streams. */
    structDelta_Tube* right1 = nullptr;  /* If null: radiation at right edge. */
    structDelta_Tube* right2 = nullptr;  /* If not null: a stream splitting into two. */
    long parallel = 1;   /* Parallel subdivision. */
    
    /* Controlled by articulation: quasistatic. */
    
    double Dxeq = 0, Dyeq = 0, Dzeq = 0;
    double mass = 0, k1 = 0, k3 = 0, Brel = 0, s1 = 0, s3 = 0, dy = 0;
    double k1left1 = 0, k1left2 = 0, k1right1 = 0, k1right2 = 0;   /* Linear coupling factors. */
    double k3left1 = 0, k3left2 = 0, k3right1 = 0, k3right2 = 0;   /* Cubic coupling factors. */
    
    /* Dynamic. */
    
    double Jhalf = 0, Jleft = 0, Jleftnew = 0, Jright = 0, Jrightnew = 0; // mass flow
    double Qhalf = 0, Qleft = 0, Qleftnew = 0, Qright = 0, Qrightnew = 0; // pressure (Qhalf seems to be the center of the tube?)
    double Dx = 0, Dxnew = 0, dDxdt = 0, dDxdtnew = 0, Dxhalf = 0;
    double Dy = 0, Dynew = 0, dDydt = 0, dDydtnew = 0;
    double Dz = 0;
    double A = 0, Ahalf = 0, Anew = 0, V = 0, Vnew = 0;
    double e = 0, ehalf = 0, eleft = 0, eleftnew = 0, eright = 0, erightnew = 0, ehalfold = 0;
    double p = 0, phalf = 0, pleft = 0, pleftnew = 0, pright = 0, prightnew = 0; // p is momentum
    double Kleft = 0, Kleftnew = 0, Kright = 0, Krightnew = 0, Pturbright = 0, Pturbrightnew = 0;
    double B = 0, r = 0, R = 0, DeltaP = 0, v = 0;
};

typedef struct structDelta_Tube* Delta_Tube; // not a huge fan of obfuscating the pointer like this...

class Delta {
public:
    int numberOfTubes;              // >= 1
    structDelta_Tube tube[MAX_NUMBER_OF_TUBES];
public:
    Delta():numberOfTubes(MAX_NUMBER_OF_TUBES){}
    Delta(int numTubes):numberOfTubes(numTubes){ assert(numTubes >= 1 && numberOfTubes <= MAX_NUMBER_OF_TUBES); }
};

/* End of file Delta.h */
#endif
