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
using namespace std;
typedef struct structDelta_Tube *Delta_Tube;

class Delta {
public:
	int numberOfTubes;              // >= 0
    vector <structDelta_Tube>  tube; // tube [0..numberOfTubes-1]
public:
    Delta();
    Delta(int numberOfTubes);
    ~Delta();
};

struct structDelta_Tube
{
    /* Structure: static. */
    Delta_Tube left1 = nullptr;   /* If null: closed at left edge. */
    Delta_Tube left2 = nullptr;   /* If not null: two merging streams. */
    Delta_Tube right1 = nullptr;  /* If null: radiation at right edge. */
    Delta_Tube right2 = nullptr;  /* If not null: a stream splitting into two. */
    long parallel = 1;   /* Parallel subdivision. */
    
    /* Controlled by articulation: quasistatic. */
    
    double Dxeq, Dyeq, Dzeq = 0;
    double mass, k1, k3, Brel, s1, s3, dy = 0;
    double k1left1, k1left2, k1right1, k1right2 = 0;   /* Linear coupling factors. */
    double k3left1, k3left2, k3right1, k3right2 = 0;   /* Cubic coupling factors. */
    
    /* Dynamic. */
    
    double Jhalf, Jleft, Jleftnew, Jright, Jrightnew = 0;
    double Qhalf, Qleft, Qleftnew, Qright, Qrightnew = 0;
    double Dx, Dxnew, dDxdt, dDxdtnew, Dxhalf = 0;
    double Dy, Dynew, dDydt, dDydtnew = 0;
    double Dz = 0;
    double A, Ahalf, Anew, V, Vnew = 0;
    double e, ehalf, eleft, eleftnew, eright, erightnew, ehalfold = 0;
    double p, phalf, pleft, pleftnew, pright, prightnew = 0;
    double Kleft, Kleftnew, Kright, Krightnew, Pturbright, Pturbrightnew = 0;
    double B, r, R, DeltaP, v = 0;
};

/* End of file Delta.h */
#endif
