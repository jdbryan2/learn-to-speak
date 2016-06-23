#ifndef _support_functions_h_
#define _support_functions_h_
//  support_funtions.h
//  vt-model
//
//  Created by Jacob D Bryan on 4/14/16.
//  Copyright © 2016 Team Jacob. All rights reserved.ctions_h_

#include "Speaker.h"
#include "Delta.h"

/* code is modified from that found in:
    #include "Articulation.h"
    #include "Art_Speaker.h"
    #include "Speaker_to_Delta.h"
*/

void Speaker_to_Delta (Speaker &me, Delta &thee);


void Art_Speaker_toVocalTract (double *art, Speaker &speaker,
    double intX [], double intY [], double extX [], double extY [],
    double *bodyX, double *bodyY);
/*
    Function:
        compute key places of the supralaryngeal vocal tract.
    Preconditions:
        index intX [1..13];
        index intY [1..13];
        index extX [1..9];
        index extY [1..9];
    Postconditions:
        int [1..6] is anterior larynx, hyoid, and tongue root.
        int [6..7] is the arc of the tongue body.
        int [7..13] is tongue blade, lower teeth, and lower lip.
        ext [1..5] is posterior larynx, back pharynx wall, and velic.
        ext [5..6] is the arc of the velum and palate.
        ext [6..9] is the gums, upper teeth and upper lip.
*/

void Art_Speaker_meshVocalTract (double *art, Speaker &speaker,
    double xi [], double yi [], double xe [], double ye [],
    double xmm [], double ymm [], int closed []);

/* TODO: Implement something to do graphics for drawing VT.
#include "Graphics.h"
void Art_Speaker_draw (Art art, Speaker speaker, Graphics g);
void Art_Speaker_fillInnerContour (Art art, Speaker speaker, Graphics g);
void Art_Speaker_drawMesh (Art art, Speaker speaker, Graphics g);
 */

/* End of file Speaker_to_Delta.h */
#endif
