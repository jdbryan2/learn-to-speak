/* Speaker_to_Delta.cpp
 *
 * Copyright (C) 1992-2011,2015,2016 Paul Boersma
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

#include "Art_Speaker.h"
#include "Speaker_to_Delta.h"
#include <assert.h>
#include <math.h>
#define SMOOTH_LUNGS  true
#define FIRST_TUBE  6

void Speaker_to_Delta (Speaker &me, Delta &thee) {
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
        Art art;
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
