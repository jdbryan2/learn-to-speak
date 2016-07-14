//
//  VocalTract.cpp
//  vt-model
//
//  Created by William J Wagner on 7/14/16.
//  Copyright © 2016 Team Jacob. All rights reserved.
//

#include "VocalTract.h"
#include <math.h>
#include <assert.h>


using namespace std;

VocalTract::VocalTract(string kindOfSpeaker, int numberOfVocalCordMasses)
{
    /* Preconditions:
     *    1 <= numberOfVocalCordMasses <= 2;
     * Failures:
     *    Kind of speaker is not one of "Female", "Male", or "Child".	*/
    
    /* Supralaryngeal dimensions are taken from P. Mermelstein (1973):
     *    "Articulatory model for the study of speech production",
     *    Journal of the Acoustical Society of America 53,1070 - 1082.
     * That was a male speaker, so we need relativeSize for other speakers:		*/
    
    /* Laryngeal system. Data for male speaker from Ishizaka and Flanagan.	*/
    
    if (kindOfSpeaker.compare("Female")==0) {
        relativeSize = 1.0;
        lowerCord.thickness = 1.4e-3;   // dx, in metres
        upperCord.thickness = 0.7e-3;
        cord.length = 10e-3;
        lowerCord.mass = 0.02e-3;   // kilograms
        upperCord.mass = 0.01e-3;
        lowerCord.k1 = 10;   // Newtons per metre
        upperCord.k1 = 4;
    } else if (kindOfSpeaker.compare("Male")==0) {
        relativeSize = 1.1;
        lowerCord.thickness = 2.0e-3;   // dx, in metres
        upperCord.thickness = 1.0e-3;
        cord.length = 18e-3;
        lowerCord.mass = 0.1e-3;   // kilograms
        upperCord.mass = 0.05e-3;
        lowerCord.k1 = 12;   // Newtons per metre
        upperCord.k1 = 4;
    } else /* "Child" */ {
        relativeSize = 0.7;
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
    
    /* Supralaryngeal system. Data from Mermelstein. */
    velum.x = -0.031 * relativeSize;
    velum.y = 0.023 * relativeSize;
    velum.a = atan2 (velum.y, velum.x);
    palate.radius = sqrt (velum.x * velum.x + velum.y * velum.y);
    tip.length = 0.034 * relativeSize;
    neutralBodyDistance = 0.086 * relativeSize;
    alveoli.x = 0.024 * relativeSize;
    alveoli.y = 0.0302 * relativeSize;
    alveoli.a = atan2 (alveoli.y, alveoli.x);
    teethCavity.dx1 = -0.009 * relativeSize;
    teethCavity.dx2 = -0.004 * relativeSize;
    teethCavity.dy = -0.011 * relativeSize;
    lowerTeeth.a = -0.30;   // radians
    lowerTeeth.r = 0.113 * relativeSize;   // metres
    upperTeeth.x = 0.036 * relativeSize;
    upperTeeth.y = 0.026 * relativeSize;
    lowerLip.dx = 0.010 * relativeSize;
    lowerLip.dy = -0.004 * relativeSize;
    upperLip.dx = 0.010 * relativeSize;
    upperLip.dy = 0.004 * relativeSize;
    
    nose.Dx = 0.007 * relativeSize;
    nose.Dz = 0.014 * relativeSize;
    nose.weq [0] = 0.018 * relativeSize;
    nose.weq [1] = 0.016 * relativeSize;
    nose.weq [2] = 0.014 * relativeSize;
    nose.weq [3] = 0.020 * relativeSize;
    nose.weq [4] = 0.023 * relativeSize;
    nose.weq [5] = 0.020 * relativeSize;
    nose.weq [6] = 0.035 * relativeSize;
    nose.weq [7] = 0.035 * relativeSize;
    nose.weq [8] = 0.030 * relativeSize;
    nose.weq [9] = 0.022 * relativeSize;
    nose.weq [10] = 0.016 * relativeSize;
    nose.weq [11] = 0.010 * relativeSize;
    nose.weq [12] = 0.012 * relativeSize;
    nose.weq [13] = 0.013 * relativeSize;
    
    shunt.Dx = 0;
    shunt.Dy = 0;
    shunt.Dz = 0;
}



/* Art_Speaker.cpp */

// TODO: These arrays (intx,inty,extx,exty)are being indexed starting @ 1.
//       this is very confusing, but it would be time consuming to switch it over.
//       we should change this at some point.
//
//       It would seem that the indexing of these variables is meant to map directly
//       to the diagram on page 53 of functional phonology. Index 15 appears to be
//       the chin (not given an index in the book) and the next vertex after the chin
//       is at the larynx so index 16 wraps around to 1.
//
//       In Speaker_to_Delta.cpp, it looks like these variables are initialized with
//       30 elements each before being passed in to the various functions. I guess the
//       first element (0th index) is left unused and so are the remaining 3 at the end?
//       This is fucking weird but if the model works, then I don't think it's worth the
//       trouble to fix.
//
void VocalTract::ArticulateUpper (double art[kArt_muscle_MAX])
{
    double f = this->relativeSize * 1e-3;
    struct { double x, y, da; } jaw;
    struct { double dx, dy; } hyoid;
    struct { double x, y, r, radius; } body;
    struct { double x, y, r, a; } teeth;
    struct { double a; } blade;
    struct { double x, y, a; } tip;
    struct { double dx, dy; } lowerLip, upperLip;
    double HBody_x, HBody_y, HC, Sp, p, a, b;
    
    /* Determine the position of the hyoid bone (Mermelstein's H).	*/
    /* The rest position is a characteristic of the speaker.		*/
    /* The stylohyoid muscle pulls the hyoid bone up.			*/
    /* The sternohyoid muscle pulls the hyoid bone down.			*/
    /* The sphincter muscle pulls the hyoid bone backwards.		*/
    
    hyoid.dx = -5 * f * art [kArt_muscle_SPHINCTER];
    hyoid.dy = 20 * f * (art [kArt_muscle_STYLOHYOID]
                         - art [kArt_muscle_STERNOHYOID]);
    
    /* The larynx moves up and down with the hyoid bone.			*/
    /* Only the lowest point (Mermelstein's K)				*/
    /* does not follow completely the horizontal movements.		*/
    
    /* Anterior larynx. */
    intX [1] = -14 * f + 0.5 * hyoid.dx;		intY [1] = -53 * f + hyoid.dy;
    /* Top of larynx. */
    intX [2] = -20 * f + hyoid.dx;		intY [2] = -33 * f + hyoid.dy;
    /* Epiglottis. */
    intX [3] = -20 * f + hyoid.dx;		intY [3] = -26 * f + hyoid.dy;
    /* Hyoid bone. */
    intX [4] = -16 * f + hyoid.dx;		intY [4] = -26 * f + hyoid.dy;
    /* Posterior larynx. */
    extX [1] = -22 * f + hyoid.dx;		extY [1] = -53 * f + hyoid.dy;
    /* Esophagus. */
    extX [2] = -26 * f + hyoid.dx;		extY [2] = -40 * f + hyoid.dy;
    
    /* The lower pharynx moves up and down with the hyoid bone.			*/
    /* The lower constrictor muscle pulls the rear pharyngeal wall forwards.	*/
    
    extX [3] = -34 * f + art [kArt_muscle_SPHINCTER] * 5 * f;	extY [3] = extY [2];
    
    /* The upper pharynx is fixed at the height of the velum. */
    /* The upper constrictor muscle pulls the rear pharyngeal wall forwards. */
    
    extX [5] = -34 * f + art [kArt_muscle_SPHINCTER] * 5 * f;
    extY [5] = this->velum.y;
    
    /* The height of the middle pharynx is in between the lower and upper pharynx. */
    /* The middle constrictor muscle pulls the rear pharyngeal wall forwards. */
    
    extX [4] = -34 * f + art [kArt_muscle_SPHINCTER] * 5 * f;
    extY [4] = 0.5 * (extY [3] + extY [5]);
    
    /* Tongue root. */
    
    jaw.x = -75 * f, jaw.y = 53 * f;   /* Position of the condyle. */
    jaw.da = art [kArt_muscle_MASSETER] * 0.15
    - art [kArt_muscle_MYLOHYOID] * 0.20;
    body.x = jaw.x + 81 * f * cos (-0.60 + jaw.da)
    - art [kArt_muscle_STYLOGLOSSUS] * 10 * f
    + art [kArt_muscle_GENIOGLOSSUS] * 10 * f;
    body.y = jaw.y + 81 * f * sin (-0.60 + jaw.da)
    - art [kArt_muscle_HYOGLOSSUS] * 10 * f
    + art [kArt_muscle_STYLOGLOSSUS] * 5 * f;
    bodyX = body.x;
    bodyY = body.y;
    body.r = sqrt ((jaw.x - body.x) * (jaw.x - body.x)
                   + (jaw.y - body.y) * (jaw.y - body.y));
    body.radius = 20 * f;
    HBody_x = body.x - intX [4];
    HBody_y = body.y - intY [4];
    HC = sqrt (HBody_x * HBody_x + HBody_y * HBody_y);
    if (HC <= body.radius) {
        HC = body.radius;
        Sp = 0.0;   // prevent rounding errors in sqrt (can occur on processors with e.g. 80-bit registers)
    } else {
        Sp = sqrt (HC * HC - body.radius * body.radius);
    }
    a = atan2 (HBody_y, HBody_x);
    b = asin (body.radius / HC);
    p = 0.57 * (34.8 * f - Sp);
    intX [5] = intX [4] + 0.5 * Sp * cos (a + b) - p * sin (a + b);
    intY [5] = intY [4] + 0.5 * Sp * sin (a + b) + p * cos (a + b);
    HBody_x = body.x - intX [5];
    HBody_y = body.y - intY [5];
    HC = sqrt (HBody_x * HBody_x + HBody_y * HBody_y);
    if (HC <= body.radius) { HC = body.radius; Sp = 0.0; } else Sp = sqrt (HC * HC - body.radius * body.radius);
    a = atan2 (HBody_y, HBody_x);
    b = asin (body.radius / HC);
    intX [6] = intX [5] + Sp * cos (a + b);
    intY [6] = intY [5] + Sp * sin (a + b);
    
    /* Posterior blade. */
    
    teeth.a = this->lowerTeeth.a + jaw.da;
    intX [7] = body.x + body.radius * cos (1.73 + teeth.a);
    intY [7] = body.y + body.radius * sin (1.73 + teeth.a);
    
    /* Tip. */
    
    tip.a = (art [kArt_muscle_UPPER_TONGUE]
             - art [kArt_muscle_LOWER_TONGUE]) * 1.0;
    blade.a = teeth.a
    + 0.004 * (body.r - this->neutralBodyDistance) + tip.a;
    intX [8] = intX [7] + this->tip.length * cos (blade.a);
    intY [8] = intY [7] + this->tip.length * sin (blade.a);
    
    /* Jaw. */
    
    teeth.r = this->lowerTeeth.r;
    teeth.x = jaw.x + teeth.r * cos (teeth.a);
    teeth.y = jaw.y + teeth.r * sin (teeth.a);
    intX [9] = teeth.x + this->teethCavity.dx1;
    intY [9] = teeth.y + this->teethCavity.dy;
    intX [10] = teeth.x + this->teethCavity.dx2;
    intY [10] = intY [9];
    intX [11] = teeth.x;
    intY [11] = teeth.y;
    
    /* Lower lip. */
    
    lowerLip.dx = this->lowerLip.dx + art [kArt_muscle_ORBICULARIS_ORIS] * 0.02 - 5e-3;
    lowerLip.dy = this->lowerLip.dy + art [kArt_muscle_ORBICULARIS_ORIS] * 0.01;
    intX [12] = teeth.x;
    intY [12] = teeth.y + lowerLip.dy;
    intX [13] = teeth.x + lowerLip.dx;
    intY [13] = intY [12];
    
    /* Velum. */
    
    extX [6] = this->velum.x;
    extY [6] = this->velum.y;
    
    /* Palate. */
    
    extX [7] = this->alveoli.x;
    extY [7] = this->alveoli.y;
    extX [8] = this->upperTeeth.x;
    extY [8] = this->upperTeeth.y;
    
    /* Upper lip. */
    
    upperLip.dx = this->upperLip.dx + art [kArt_muscle_ORBICULARIS_ORIS] * 0.02 - 5e-3;
    upperLip.dy = this->upperLip.dy - art [kArt_muscle_ORBICULARIS_ORIS] * 0.01;
    extX [9] = extX [8];
    extY [9] = extY [8] + upperLip.dy;
    extX [10] = extX [9] + upperLip.dx;
    extY [10] = extY [9];
    extX [11] = extX [10] + 5e-3;
    extY [11] = extY [10] + DLIP;
    
    /* Chin. */
    
    intX [14] = intX [13] + 5e-3;
    intY [14] = intY [13] - DLIP;
    intX [15] = intX [11] + 0.5e-2;
    intY [15] = intY [11] - 3.0e-2;
    intX [16] = intX [1];
    intY [16] = intY [1];
}


// TODO: These arrays (xmm,ymm,xi,yi,xe,ye,dx,dy)are being indexed starting @ 1.
//       This is very confusing, but it would be time consuming to switch it over.
//       We should change this at some point.
void VocalTract::MeshUpper (double art[kArt_muscle_MAX])
{
    double f = relativeSize * 1e-3;
    double d_angle = 0.0;
    int i;
    
    ArticulateUpper(art);
    bodyRadius = 20 * f;
    
    xe [1] = extX [1];   /* Eq. 5.45 */
    ye [1] = extY [1];
    xe [2] = 0.2 * extX [2] + 0.8 * extX [1];
    ye [2] = 0.2 * extY [2] + 0.8 * extY [1];
    xe [3] = 0.6 * extX [2] + 0.4 * extX [1];
    ye [3] = 0.6 * extY [2] + 0.4 * extY [1];
    xe [4] = 0.9 * extX [3] + 0.1 * extX [4];   /* Eq. 5.46 */
    ye [4] = 0.9 * extY [3] + 0.1 * extY [4];
    xe [5] = 0.7 * extX [3] + 0.3 * extX [4];
    ye [5] = 0.7 * extY [3] + 0.3 * extY [4];
    xe [6] = 0.5 * extX [3] + 0.5 * extX [4];
    ye [6] = 0.5 * extY [3] + 0.5 * extY [4];
    xe [7] = 0.3 * extX [3] + 0.7 * extX [4];
    ye [7] = 0.3 * extY [3] + 0.7 * extY [4];
    xe [8] = 0.1 * extX [3] + 0.9 * extX [4];
    ye [8] = 0.1 * extY [3] + 0.9 * extY [4];
    xe [9] = 0.9 * extX [4] + 0.1 * extX [5];
    ye [9] = 0.9 * extY [4] + 0.1 * extY [5];
    xe [10] = 0.7 * extX [4] + 0.3 * extX [5];
    ye [10] = 0.7 * extY [4] + 0.3 * extY [5];
    xe [11] = 0.5 * extX [4] + 0.5 * extX [5];
    ye [11] = 0.5 * extY [4] + 0.5 * extY [5];
    xe [12] = 0.3 * extX [4] + 0.7 * extX [5];
    ye [12] = 0.3 * extY [4] + 0.7 * extY [5];
    xe [13] = 0.1 * extX [4] + 0.9 * extX [5];
    ye [13] = 0.1 * extY [4] + 0.9 * extY [5];
    d_angle = (atan2 (ye [13], xe [13]) - 0.5 * M_PI) / 6;   /* Eq. 5.47 */
    for (i = 14; i <= 18; i ++) {
        double a = 0.5 * M_PI + (19 - i) * d_angle;
        xe [i] = palate.radius * cos (a);
        ye [i] = palate.radius * sin (a);
    }
    xe [19] = 0;
    ye [19] = palate.radius;
    xe [20] = 0.25 * extX [7];
    xe [21] = 0.50 * extX [7];
    xe [22] = 0.75 * extX [7];
    for (i = 20; i <= 22; i ++) {
        ye [i] = palate.radius * sqrt (1.0 - xe [i] * xe [i] /
                                               (palate.radius * palate.radius));
    }
    xe [23] = extX [7];
    ye [23] = extY [7];
    xe [24] = 0.5 * (extX [7] + extX [8]);
    ye [24] = 0.5 * (extY [7] + extY [8]);
    xe [25] = extX [8];
    ye [25] = extY [8];
    xe [26] = 0.25 * extX [11] + 0.75 * extX [9];
    xe [27] = 0.75 * extX [11] + 0.25 * extX [9];
    ye [26] = extY [10];
    ye [27] = 0.5 * (extY [10] + extY [11]);
    for (i = 1; i <= ART_SPEAKER_MESHCOUNT; i ++) {   /* Every mesh point. */
        double minimum = 100000;
        int j;
        for (j = 1; j <= 15 - 1; j ++) {   /* Every internal segment. */
            double d = toLine (xe [i], ye [i], j);
            if (d < minimum) minimum = d;
        }
        if ((closed [i] = inside (xe [i], ye [i])) != 0)
            minimum = - minimum;
        if (xe [i] >= 0.0) {   /* Vertical line pieces. */
            xi [i] = xe [i];
            yi [i] = ye [i] - minimum;
        } else if (ye [i] <= 0.0) {   /* Horizontal line pieces. */
            xi [i] = xe [i] + minimum;
            yi [i] = ye [i];
        } else {   /* Radial line pieces, centre = centre of palate arc. */
            double angle = atan2 (ye [i], xe [i]);
            xi [i] = xe [i] - minimum * cos (angle);
            yi [i] = ye [i] - minimum * sin (angle);
        }
    }
    for (i = 1; i <= ART_SPEAKER_MESHCOUNT; i ++) {
        xm [i] = 0.5 * (xe [i] + xi [i]);
        ym [i] = 0.5 * (ye [i] + yi [i]);
    }
    for (i = 2; i <= ART_SPEAKER_MESHCOUNT; i ++) {
        xmm [i] = 0.5 * (xm [i - 1] + xm [i]);
        ymm [i] = 0.5 * (ym [i - 1] + ym [i]);
    }
    xmm [1] = 2 * xm [1] - xmm [2];
    ymm [1] = 2 * ym [1] - ymm [2];
    xmm [ART_SPEAKER_MESHCOUNT + 1] = 2 * xm [ART_SPEAKER_MESHCOUNT]
    - xmm [ART_SPEAKER_MESHCOUNT];
    ymm [ART_SPEAKER_MESHCOUNT + 1] = 2 * ym [ART_SPEAKER_MESHCOUNT]
    - ymm [ART_SPEAKER_MESHCOUNT];
}

double VocalTract::arcLength (double from, double to) {
    double result = to - from;
    while (result > 0.0) result -= 2 * M_PI;
    while (result < 0.0) result += 2 * M_PI;
    return result;
}

double VocalTract::toLine (double x, double y, int i) {
    int nearby;
    if (i == 6) {
        double a7 = atan2 (intY [7] - bodyY, intX [7] - bodyX);
        double a6 = atan2 (intY [6] - bodyY, intX [6] - bodyX);
        double a = atan2 (y - bodyY, x - bodyX);
        double da6 = arcLength (a7, a6);
        double da = arcLength (a7, a);
        if (da <= da6)
            return fabs (sqrt ((bodyX - x) * (bodyX - x) + (bodyY - y) * (bodyY - y)) - bodyRadius);
        else
            nearby = arcLength (a7 + 0.5 * da6, a) < M_PI ? 6 : 7;
    } else if ((x - intX [i]) * (intX [i + 1] - intX [i]) +
               (y - intY [i]) * (intY [i + 1] - intY [i]) < 0) {
        nearby = i;
    } else if ((x - intX [i + 1]) * (intX [i] - intX [i + 1]) +
               (y - intY [i + 1]) * (intY [i] - intY [i + 1]) < 0) {
        nearby = i + 1;
    } else {
        double boundaryDistance =
        sqrt ((intX [i + 1] - intX [i]) * (intX [i + 1] - intX [i]) +
              (intY [i + 1] - intY [i]) * (intY [i + 1] - intY [i]));
        double outerProduct = (intX [i] - x) * (intY [i + 1] - intY [i]) - (intY [i] - y) * (intX [i + 1] - intX [i]);
        return fabs (outerProduct) / boundaryDistance;
    }
    return sqrt ((intX [nearby] - x) * (intX [nearby] - x) + (intY [nearby] - y) * (intY [nearby] - y));
}

int VocalTract::inside (double x, double y)
{
    int i, up = 0;
    for (i = 1; i <= 16 - 1; i ++)
        if ((y > intY [i]) != (y > intY [i + 1])) {
            double slope = (intX [i + 1] - intX [i]) / (intY [i + 1] - intY [i]);
            if (x > intX [i] + (y - intY [i]) * slope)
                up += ( y > intY [i] ? 1 : -1 );
        }
    return up != 0 || bodyRadius * bodyRadius >
    (x - bodyX) * (x - bodyX) + (y - bodyY) * (y - bodyY);
}


// TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
//       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
double VocalTract::MeshSumX(int i) {
    double dx = 0, dy = 0;
    return sqrt (( dx = xmm [i] - xmm [i + 1], dx * dx ) + ( dy = ymm [i] - ymm [i + 1], dy * dy ));
}

// TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
//       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
double VocalTract::MeshSumY(int i) {
    double dx = 0, dy = 0;
    return sqrt (( dx = xe [i] - xi [i], dx * dx ) + ( dy = ye [i] - yi [i], dy * dy ));
}