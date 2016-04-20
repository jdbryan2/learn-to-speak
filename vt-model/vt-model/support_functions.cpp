
//  support_funtions.cpp
//  vt-model
//
//  Created by Jacob D Bryan on 4/14/16.
//  Copyright © 2016 Team Jacob. All rights reserved.

#include "support_functions.h"
#include <assert.h>
#include <math.h>
#define SMOOTH_LUNGS  true
#define FIRST_TUBE  6
#define DLIP  5e-3

void Speaker_to_Delta (Speaker &me, Delta &thee) 
{
	double f = me.relativeSize * 1e-3;   // we shall use millimetres and grams
	double xe [30], ye [30], xi [30], yi [30], xmm [30], ymm [30], dx, dy;
	int closed [40];
	int itube;
	assert(me.cord.numberOfMasses == 1 || me.cord.numberOfMasses == 2 || me.cord.numberOfMasses == 10);
    assert(thee.numberOfTubes == 89);
	/* Lungs: tubes 0..22. */

	for (itube = 0; itube <= 22; itube ++) {
        Delta_Tube t = &(thee.tube[itube]);
		t -> Dx = t -> Dxeq = 10.0 * f;
		t -> Dy = t -> Dyeq = 100.0 * f;
		t -> Dz = t -> Dzeq = 230.0 * f;
		t -> mass = 10.0 * me.relativeSize * t -> Dx * t -> Dz;   // 80 * f; 35 * Dx * Dz
		t -> k1 = 200.0;   // 90000 * Dx * Dz; Newtons per metre
		t -> k3 = 0.0;
		t -> Brel = 0.8;
		t -> parallel = 1000;
	}

	/* Bronchi: tubes 23..28. */

	for (itube = 23; itube <= 28; itube ++) {
		Delta_Tube t = &(thee.tube[itube]);
		t -> Dx = t -> Dxeq = 10.0 * f;
		t -> Dy = t -> Dyeq = 15.0 * f;
		t -> Dz = t -> Dzeq = 30.0 * f;
		t -> mass = 10.0 * f;
		t -> k1 = 40.0;   // 125000 * Dx * Dz; Newtons per metre
		t -> k3 = 0.0;
		t -> Brel = 0.8;
	}

	/* Trachea: tubes 29..34; four of these may be replaced by conus elasticus (see below). */

	for (itube = 29; itube <= 34; itube ++) {
        Delta_Tube t = &(thee.tube[itube]);
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
			Delta_Tube t = &(thee.tube[data[i].itube]);
			t -> Dy = t -> Dyeq = data [i]. Dy * f;
			t -> Dz = t -> Dzeq = data [i]. Dz * f;
			t -> parallel = data [i]. parallel;
		}
		for (itube = 25; itube <= 34; itube ++) {
			Delta_Tube t = &(thee.tube[itube]);
			t -> Dy = t -> Dyeq = 11.0 * f;
			t -> Dz = t -> Dzeq = 14.0 * f;
			t -> parallel = 1;
		}
		for (itube = FIRST_TUBE; itube <= 17; itube ++) {
			Delta_Tube t = &(thee.tube[itube]);
			t -> Dx = t -> Dxeq = 10.0 * f;
			t -> mass = 10.0 * me.relativeSize * t -> Dx * t -> Dz;   // 10 mm
			t -> k1 = 1e5 * t -> Dx * t -> Dz;   // elastic tissue: 1 mbar/mm
			t -> k3 = 0.0;
			t -> Brel = 1.0;
		}
		for (itube = 18; itube <= 34; itube ++) {
			Delta_Tube t = &(thee.tube[itube]);
			t -> Dx = t -> Dxeq = 10.0 * f;
			t -> mass = 3.0 * me.relativeSize * t -> Dx * t -> Dz;   // 3 mm
			t -> k1 = 10e5 * t -> Dx * t -> Dz;   // cartilage: 10 mbar/mm
			t -> k3 = 0.0;
			t -> Brel = 1.0;
		}
	}

	/* Glottis: tubes 35 and 36; the last one may be disconnected (see below). */
	{
		Delta_Tube t = &(thee.tube[35]);
		t -> Dx = t -> Dxeq = me.lowerCord.thickness;
		t -> Dy = t -> Dyeq = 0.0;
		t -> Dz = t -> Dzeq = me.cord.length;
		t -> mass = me.lowerCord.mass;
		t -> k1 = me.lowerCord.k1;
		t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
		t -> Brel = 0.2;
	}

	/*
	 * Fill in the values for the upper part of the glottis (tube 36) only if there is no one-mass model.
	 */
	if (me.cord.numberOfMasses >= 2) {
		Delta_Tube t = &(thee.tube[36]);
		t -> Dx = t -> Dxeq = me.upperCord.thickness;
		t -> Dy = t -> Dyeq = 0.0;
		t -> Dz = t -> Dzeq = me.cord.length;
		t -> mass = me.upperCord.mass;
		t -> k1 = me.upperCord.k1;
		t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
		t -> Brel = 0.2;

		/* Couple spring with lower cord. */
		t -> k1left1 = thee.tube[35].k1right1 = 1.0;
	}

	/*
	 * Fill in the values for the conus elasticus (tubes 78..85) only if we want to model it.
	 */
	if (me.cord.numberOfMasses == 10) {
		thee.tube[78].Dx = thee.tube[78]. Dxeq = 8.0 * f;
		thee.tube[79].Dx = thee.tube[79].Dxeq = 7.0 * f;
		thee.tube[80].Dx = thee.tube[80].Dxeq = 6.0 * f;
		thee.tube[81].Dx = thee.tube[81].Dxeq = 5.0 * f;
		thee.tube[82].Dx = thee.tube[82].Dxeq = 4.0 * f;
		thee.tube[83].Dx = thee.tube[83].Dxeq = 0.75 * 4.0 * f + 0.25 * me.lowerCord.thickness;
		thee.tube[84].Dx = thee.tube[84].Dxeq = 0.50 * 4.0 * f + 0.50 * me.lowerCord.thickness;
		thee.tube[85].Dx = thee.tube[85].Dxeq = 0.25 * 4.0 * f + 0.75 * me.lowerCord.thickness;

		thee.tube[78].Dy = thee.tube[78].Dyeq = 11.0 * f;
		thee.tube[79].Dy = thee.tube[79].Dyeq = 7.0 * f;
		thee.tube[80].Dy = thee.tube[80].Dyeq = 4.0 * f;
		thee.tube[81].Dy = thee.tube[81].Dyeq = 2.0 * f;
		thee.tube[82].Dy = thee.tube[82].Dyeq = 1.0 * f;
		thee.tube[83].Dy = thee.tube[83].Dyeq = 0.75 * f;
		thee.tube[84].Dy = thee.tube[84].Dyeq = 0.50 * f;
		thee.tube[85].Dy = thee.tube[85].Dyeq = 0.25 * f;

		thee.tube[78].Dz = thee.tube[78].Dzeq = 16.0 * f;
		thee.tube[79].Dz = thee.tube[79].Dzeq = 16.0 * f;
		thee.tube[80].Dz = thee.tube[80].Dzeq = 16.0 * f;
		thee.tube[81].Dz = thee.tube[81].Dzeq = 16.0 * f;
		thee.tube[82].Dz = thee.tube[82].Dzeq = 16.0 * f;
		thee.tube[83].Dz = thee.tube[83].Dzeq = 0.75 * 16.0 * f + 0.25 * me.cord.length;
		thee.tube[84].Dz = thee.tube[84].Dzeq = 0.50 * 16.0 * f + 0.50 * me.cord.length;
		thee.tube[85].Dz = thee.tube[85].Dzeq = 0.25 * 16.0 * f + 0.75 * me.cord.length;

		thee.tube[78].k1 = 160.0;
		thee.tube[79].k1 = 160.0;
		thee.tube[80].k1 = 160.0;
		thee.tube[81].k1 = 160.0;
		thee.tube[82].k1 = 160.0;
		thee.tube[83].k1 = 0.75 * 160.0 * f + 0.25 * me.lowerCord.k1;
		thee.tube[84].k1 = 0.50 * 160.0 * f + 0.50 * me.lowerCord.k1;
		thee.tube[85].k1 = 0.25 * 160.0 * f + 0.75 * me.lowerCord.k1;

		thee.tube[78].Brel = 0.7;
		thee.tube[79].Brel = 0.6;
		thee.tube[80].Brel = 0.5;
		thee.tube[81].Brel = 0.4;
		thee.tube[82].Brel = 0.3;
		thee.tube[83].Brel = 0.2;
		thee.tube[84].Brel = 0.2;
		thee.tube[85].Brel = 0.2;

		for (itube = 78; itube <= 85; itube ++) {
			Delta_Tube t = &(thee.tube[itube]);
			t -> mass = t -> Dx * t -> Dz / (30.0 * f);
			t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
			t -> k1left1 = t -> k1right1 = 1.0;
		}
		thee.tube[78].k1left1 = 0.0;
		thee.tube[35].k1left1 = 1.0;   // the essence: couple spring with lower vocal cords
	}

	/*
	 * Fill in the values of the glottal shunt only if we want to model it.
	 */
	if (me.shunt.Dx != 0.0) {
		for (itube = 86; itube <= 88; itube ++) {
			Delta_Tube t = &(thee.tube[itube]);
			t -> Dx = t -> Dxeq = me.shunt.Dx;
			t -> Dy = t -> Dyeq = me.shunt.Dy;
			t -> Dz = t -> Dzeq = me.shunt.Dz;
			t -> mass = 3.0 * me.upperCord.mass;   // heavy...
			t -> k1 = 3.0 * me.upperCord.k1;   // ...and stiff...
			t -> k3 = t -> k1 * (20.0 / t -> Dz) * (20.0 / t -> Dz);
			t -> Brel = 3.0;   // ...and inelastic, so that the walls will not vibrate
		}
	}

	/* Vocal tract from neutral articulation. */
	{
        double art[kArt_muscle_MAX]={}; // all values are defaulted to zero  
		Art_Speaker_meshVocalTract (art, me, xi, yi, xe, ye, xmm, ymm, closed);
	}

	/* Pharynx and mouth: tubes 37..63. */

	for (itube = 37; itube <= 63; itube ++) {
		Delta_Tube t = &(thee.tube[itube]);
		int i = itube - 36;
        // TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
        //       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
		t -> Dx = t -> Dxeq = sqrt (( dx = xmm [i] - xmm [i + 1], dx * dx ) + ( dy = ymm [i] - ymm [i + 1], dy * dy ));
		t -> Dyeq = sqrt (( dx = xe [i] - xi [i], dx * dx ) + ( dy = ye [i] - yi [i], dy * dy ));
		if (closed [i]) t -> Dyeq = - t -> Dyeq;
		t -> Dy = t -> Dyeq;
		t -> Dz = t -> Dzeq = 0.015;
		t -> mass = 0.006;
		t -> k1 = 30.0;
		t -> k3 = 0.0;
		t -> Brel = 1.0;
	}
  /* For tongue-tip vibration [r]:  thee.tube [59]. Brel = 0.1; thee.tube [59]. k1 = 3; */

	/* Nose: tubes 64..77. */

	for (itube = 64; itube <= 77; itube ++) {
		Delta_Tube t = &(thee.tube[itube]);
		t -> Dx = t -> Dxeq = me.nose.Dx;
		t -> Dy = t -> Dyeq = me.nose.weq [itube - 64]; // Zero indexing nose array
		t -> Dz = t -> Dzeq = me.nose.Dz;
		t -> mass = 0.006;
		t -> k1 = 100.0;
		t -> k3 = 0.0;
		t -> Brel = 1.0;
	}
	thee.tube[64].Dy = thee.tube[64].Dyeq = 0.0;   // override: nasopharyngeal port closed

	/* The default structure:
	 * every tube is connected on the left to the previous tube (index one lower).
	 * This corresponds to a two-mass model of the vocal cords without shunt.
	 */
	for (itube = SMOOTH_LUNGS ? FIRST_TUBE : 0; itube < thee.numberOfTubes; itube ++) {
		Delta_Tube t = &(thee.tube[itube]);
		t -> s1 = 5e6 * t -> Dx * t -> Dz;
		t -> s3 = t -> s1 / (0.9e-3 * 0.9e-3);
		t -> dy = 1e-5;
		t -> left1 = &(thee.tube[itube-1]);   // connect to the previous tube on the left
		t -> right1 = &(thee.tube[itube+1]);   // connect to the next tube on the right
	}

	/***** Connections: boundaries and interfaces. *****/

	/* The leftmost boundary: the diaphragm (tube 1).
	 * Disconnect on the left.
	 */
	thee.tube[SMOOTH_LUNGS ? FIRST_TUBE : 0]. left1 = nullptr;   // closed at diaphragm

	/* Optional one-mass model of the vocal cords.
	 * Short-circuit over tube 37 (upper glottis).
	 */
	if (me.cord.numberOfMasses == 1) {

		/* Connect the right side of tube 35 to the left side of tube 37. */
        thee.tube[35]. right1 = &(thee.tube[37]);
		thee.tube[37]. left1 = &(thee.tube[35]);

		/* Disconnect tube 36 on both sides. */
		thee.tube[36].left1 = thee.tube[36].right1 = nullptr;
	}

	/* Optionally couple vocal cords with conus elasticus.
	 * Replace tubes 31..34 (upper trachea) by tubes 78..85 (conus elasticus).
	 */
	if (me.cord.numberOfMasses == 10) {

		/* Connect the right side of tube 30 to the left side of tube 78. */
		thee.tube[30].right1 = &(thee.tube[78]);
		thee.tube[78].left1 = &(thee.tube[30]);

		/* Connect the right side of tube 85 to the left side of tube 35. */
		thee.tube[85].right1 = &(thee.tube[35]);
		thee.tube[35].left1 = &(thee.tube[85]);

		/* Disconnect tubes 31..34 on both sides. */
		thee.tube[31].left1 = thee.tube[31].right1 = nullptr;
		thee.tube[32].left1 = thee.tube[32].right1 = nullptr;
		thee.tube[33].left1 = thee.tube[33].right1 = nullptr;
		thee.tube[34].left1 = thee.tube[34].right1 = nullptr;
	} else {

		/* Disconnect tubes 78..85 on both sides. */
		for (itube = 78; itube <= 85; itube ++)
			thee.tube[itube].left1 = thee.tube[itube].right1 = nullptr;
	}

	/* Optionally add a shunt parallel to the glottis.
	 * Create a side branch from tube 33/34 (or 84/85) to tube 37/38 with tubes 86..88.
	 */
	if (me.shunt.Dx != 0.0) {
		int topOfTrachea = ( me.cord.numberOfMasses == 10 ? 85 : 34 );

		/* Create a three-way interface below the shunt.
		 * Connect lowest shunt tube (87) with top of trachea (33/34 or 84/85).
		 */
		thee.tube[topOfTrachea - 1].right2 = &(thee.tube[86]);   // trachea to shunt
		thee.tube[86].left1 = &(thee.tube[topOfTrachea - 1]);   // shunt to trachea
		thee.tube[86].Dxeq = thee.tube[topOfTrachea - 1].Dxeq = thee.tube[topOfTrachea].Dxeq;   // equal length
		thee.tube[86].Dx = thee.tube[topOfTrachea - 1].Dx = thee.tube[topOfTrachea].Dx;

		/* Create a three-way interface above the shunt.
		 * Connect highest shunt tube (88) with bottom of pharynx (37/38).
		 */
		thee.tube[88].right1 = &(thee.tube[38]);   // shunt to pharynx
		thee.tube[38].left2 = &(thee.tube[88]);   // pharynx to shunt
		thee.tube[88].Dxeq = thee.tube[38].Dxeq = thee.tube[37].Dxeq;   // all three of equal length
		thee.tube[88].Dx = thee.tube[38].Dx = thee.tube[37].Dx;
	} else {

		/* Disconnect tubes 86..88 on both sides. */
		for (itube = 86; itube <= 88; itube ++)
			thee.tube[itube].left1 = thee.tube[itube].right1 = nullptr;
	}

	/* Create a three-way interface at the nasopharyngeal port.
	 * Connect tubes 49 (pharynx), 50 (mouth), and 64 (nose).
	 */
	thee.tube[49].right2 = &(thee.tube[64]);   // pharynx to nose
	thee.tube[64].left1 = &(thee.tube[49]);   // nose to pharynx
	thee.tube[64].Dxeq = thee.tube[50].Dxeq = thee.tube[49].Dxeq;   // all three must be of equal length
	thee.tube[64].Dx = thee.tube[50].Dx = thee.tube[49].Dx;

	/* The rightmost boundaries: the lips (tube 63) and the nostrils (tube 77).
	 * Disconnect on the right.
	 */
	thee.tube[63]. right1 = nullptr;   // radiation at the lips
	thee.tube[77]. right1 = nullptr;   // radiation at the nostrils

	for (itube = 0; itube < thee.numberOfTubes; itube ++) {
		Delta_Tube t = &(thee.tube[itube]);
		assert(! t->left1 || t->left1->right1 == t || t->left1->right2 == t);
		assert(! t->left2 || t->left2->right1 == t);
		assert(! t->right1 || t->right1->left1 == t || t->right1->left2 == t);
		assert(! t->right2 || t->right2->left1 == t);
	}
}
/* End of file Speaker_to_Delta.cpp */


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
void Art_Speaker_toVocalTract (double *art, Speaker &speaker,
	double intX [], double intY [], double extX [], double extY [],
	double *bodyX, double *bodyY)
{
	double f = speaker.relativeSize * 1e-3;
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
	extY [5] = speaker.velum.y;

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
	*bodyX = body.x;
	*bodyY = body.y;
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

	teeth.a = speaker.lowerTeeth.a + jaw.da;
	intX [7] = body.x + body.radius * cos (1.73 + teeth.a);
	intY [7] = body.y + body.radius * sin (1.73 + teeth.a);

	/* Tip. */

	tip.a = (art [kArt_muscle_UPPER_TONGUE]
		- art [kArt_muscle_LOWER_TONGUE]) * 1.0;
	blade.a = teeth.a
		+ 0.004 * (body.r - speaker.neutralBodyDistance) + tip.a;
	intX [8] = intX [7] + speaker.tip.length * cos (blade.a);
	intY [8] = intY [7] + speaker.tip.length * sin (blade.a);

	/* Jaw. */

	teeth.r = speaker.lowerTeeth.r;
	teeth.x = jaw.x + teeth.r * cos (teeth.a);
	teeth.y = jaw.y + teeth.r * sin (teeth.a);
	intX [9] = teeth.x + speaker.teethCavity.dx1;
	intY [9] = teeth.y + speaker.teethCavity.dy;
	intX [10] = teeth.x + speaker.teethCavity.dx2;
	intY [10] = intY [9];
	intX [11] = teeth.x;
	intY [11] = teeth.y;

	/* Lower lip. */

	lowerLip.dx = speaker.lowerLip.dx + art [kArt_muscle_ORBICULARIS_ORIS] * 0.02 - 5e-3;
	lowerLip.dy = speaker.lowerLip.dy + art [kArt_muscle_ORBICULARIS_ORIS] * 0.01;
	intX [12] = teeth.x;
	intY [12] = teeth.y + lowerLip.dy;
	intX [13] = teeth.x + lowerLip.dx;
	intY [13] = intY [12];

	/* Velum. */

	extX [6] = speaker.velum.x;
	extY [6] = speaker.velum.y;

	/* Palate. */

	extX [7] = speaker.alveoli.x;
	extY [7] = speaker.alveoli.y;
	extX [8] = speaker.upperTeeth.x;
	extY [8] = speaker.upperTeeth.y;

	/* Upper lip. */

	upperLip.dx = speaker.upperLip.dx + art [kArt_muscle_ORBICULARIS_ORIS] * 0.02 - 5e-3;
	upperLip.dy = speaker.upperLip.dy - art [kArt_muscle_ORBICULARIS_ORIS] * 0.01;
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

static double arcLength (double from, double to) {
	double result = to - from;
	while (result > 0.0) result -= 2 * M_PI;
	while (result < 0.0) result += 2 * M_PI;
	return result;
}

static int Art_Speaker_meshCount = 27;
static double bodyX, bodyY, bodyRadius;

static double toLine (double x, double y, const double intX [], const double intY [], int i) {
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

static int inside (double x, double y,
	const double intX [], const double intY [])
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

// TODO: These arrays (xmm,ymm,xi,yi,xe,ye,dx,dy)are being indexed starting @ 1.
//       This is very confusing, but it would be time consuming to switch it over.
//       We should change this at some point.
void Art_Speaker_meshVocalTract (double *art, Speaker &speaker,
	double xi [], double yi [], double xe [], double ye [],
	double xmm [], double ymm [], int closed [])
{
	double f = speaker.relativeSize * 1e-3;
	double intX [1 + 16], intY [1 + 16], extX [1 + 11], extY [1 + 11], d_angle;
	double xm [40], ym [40];
	int i;

	Art_Speaker_toVocalTract (art, speaker, intX, intY, extX, extY, & bodyX, & bodyY);
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
		xe [i] = speaker.palate.radius * cos (a);
		ye [i] = speaker.palate.radius * sin (a);
	}
	xe [19] = 0;
	ye [19] = speaker.palate.radius;
	xe [20] = 0.25 * extX [7];
	xe [21] = 0.50 * extX [7];
	xe [22] = 0.75 * extX [7];
	for (i = 20; i <= 22; i ++) {
		ye [i] = speaker.palate.radius * sqrt (1.0 - xe [i] * xe [i] /
			(speaker.palate.radius * speaker.palate.radius));
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
	for (i = 1; i <= 27; i ++) {   /* Every mesh point. */
		double minimum = 100000;
		int j;
		for (j = 1; j <= 15 - 1; j ++) {   /* Every internal segment. */
			double d = toLine (xe [i], ye [i], intX, intY, j);
			if (d < minimum) minimum = d;
		}
		if ((closed [i] = inside (xe [i], ye [i], intX, intY)) != 0)
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
	for (i = 1; i <= Art_Speaker_meshCount; i ++) {
		xm [i] = 0.5 * (xe [i] + xi [i]);
		ym [i] = 0.5 * (ye [i] + yi [i]);
	}
	for (i = 2; i <= Art_Speaker_meshCount; i ++) {
		xmm [i] = 0.5 * (xm [i - 1] + xm [i]);
		ymm [i] = 0.5 * (ym [i - 1] + ym [i]);
	}
	xmm [1] = 2 * xm [1] - xmm [2];
	ymm [1] = 2 * ym [1] - ymm [2];
	xmm [Art_Speaker_meshCount + 1] = 2 * xm [Art_Speaker_meshCount]
		- xmm [Art_Speaker_meshCount];
	ymm [Art_Speaker_meshCount + 1] = 2 * ym [Art_Speaker_meshCount]
		- ymm [Art_Speaker_meshCount];
}

/* TODO: Implement something to do graphics for drawing VT.
void Art_Speaker_draw (Art art, Speaker speaker, Graphics g) {
    double f = speaker.relativeSize * 1e-3;
    double intX [1 + 16], intY [1 + 16], extX [1 + 11], extY [1 + 11];
    double bodyX, bodyY;
    int i;
    Graphics_Viewport previous;
    
    Art_Speaker_toVocalTract (art, speaker, intX, intY, extX, extY, & bodyX, & bodyY);
    previous = Graphics_insetViewport (g, 0.1, 0.9, 0.1, 0.9);
    Graphics_setWindow (g, -0.05, 0.05, -0.05, 0.05);
    
    // Draw inner contour.
    
    for (i = 1; i <= 5; i ++)
        Graphics_line (g, intX [i], intY [i], intX [i + 1], intY [i + 1]);
    Graphics_arc (g, bodyX, bodyY, 20 * f,
                  atan2 (intY [7] - bodyY, intX [7] - bodyX) * 180 / NUMpi,
                  atan2 (intY [6] - bodyY, intX [6] - bodyX) * 180 / NUMpi);
    for (i = 7; i <= 15; i ++)
        Graphics_line (g, intX [i], intY [i], intX [i + 1], intY [i + 1]);
    
    // Draw outer contour.
    
    for (i = 1; i <= 5; i ++)
        Graphics_line (g, extX [i], extY [i], extX [i + 1], extY [i + 1]);
    Graphics_arc (g, 0, 0, speaker.palate.radius,
                  speaker.alveoli.a * 180 / NUMpi,
                  speaker.velum.a * 180 / NUMpi);
    for (i = 7; i <= 10; i ++)
        Graphics_line (g, extX [i], extY [i], extX [i + 1], extY [i + 1]);
    Graphics_resetViewport (g, previous);
}

void Art_Speaker_fillInnerContour (Art art, Speaker speaker, Graphics g) {
    double f = speaker.relativeSize * 1e-3;
    double intX [1 + 16], intY [1 + 16], extX [1 + 11], extY [1 + 11];
    double x [1 + 16], y [1 + 16];
    double bodyX, bodyY;
    int i;
    Graphics_Viewport previous;
    
    Art_Speaker_toVocalTract (art, speaker, intX, intY, extX, extY, & bodyX, & bodyY);
    previous = Graphics_insetViewport (g, 0.1, 0.9, 0.1, 0.9);
    Graphics_setWindow (g, -0.05, 0.05, -0.05, 0.05);
    for (i = 1; i <= 16; i ++) { x [i] = intX [i]; y [i] = intY [i]; }
    Graphics_setGrey (g, 0.8);
    Graphics_fillArea (g, 16, & x [1], & y [1]);
    Graphics_fillCircle (g, bodyX, bodyY, 20 * f);
    Graphics_setGrey (g, 0.0);
    Graphics_resetViewport (g, previous);
}

void Art_Speaker_drawMesh (Art art, Speaker speaker, Graphics graphics) {
	double xi [40], yi [40], xe [40], ye [40], xmm [40], ymm [40];
	int closed [40];
	int i;
	Graphics_Viewport previous;
	int oldLineType = Graphics_inqLineType (graphics);
	Art_Speaker_meshVocalTract (art, speaker, xi, yi, xe, ye, xmm, ymm, closed);
	previous = Graphics_insetViewport (graphics, 0.1, 0.9, 0.1, 0.9);   // Must be square.
	Graphics_setWindow (graphics, -0.05, 0.05, -0.05, 0.05);

	// Mesh lines.
	for (i = 1; i <= Art_Speaker_meshCount; i ++)
		Graphics_line (graphics, xi [i], yi [i], xe [i], ye [i]);

	// Radii.
	Graphics_setLineType (graphics, Graphics_DOTTED);
	for (i = 1; i <= Art_Speaker_meshCount; i ++)
		if (xe [i] <= 0.0 && ye [i] >= 0.0)
			Graphics_line (graphics, 0.0, 0.0, 0.9 * xi [i], 0.9 * yi [i]);
	Graphics_setLineType (graphics, oldLineType);

	// Lengths.
	for (i = 1; i <= Art_Speaker_meshCount; i ++)
		Graphics_line (graphics, xmm [i], ymm [i], xmm [i + 1], ymm [i + 1]);

	for (i = 1; i <= Art_Speaker_meshCount + 1; i ++)
		Graphics_speckle (graphics, xmm [i], ymm [i]);
	Graphics_setTextAlignment (graphics, Graphics_LEFT, Graphics_HALF);
	Graphics_text (graphics, 0.0, 0.0, U"O");   // origin
	Graphics_resetViewport (graphics, previous);
}
*/
