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
using namespace std;

Speaker::Speaker(string kindOfSpeaker, int numberOfVocalCordMasses) {
    /* Preconditions:								*/
    /*    1 <= numberOfVocalCordMasses <= 2;					*/
    /* Failures:									*/
    /*    Kind of speaker is not one of "Female", "Male", or "Child".		*/

	/* Supralaryngeal dimensions are taken from P. Mermelstein (1973):		*/
	/*    "Articulatory model for the study of speech production",		*/
	/*    Journal of the Acoustical Society of America 53,1070 - 1082.		*/
	/* That was a male speaker, so we need scaling for other speakers:		*/

	double scaling;
	if (kindOfSpeaker.compare("Male")) relativeSize = 1.1;
	else if (kindOfSpeaker.compare("Child")) relativeSize = 0.7;
	else relativeSize = 1.0;
	scaling = relativeSize;

	/* Laryngeal system. Data for male speaker from Ishizaka and Flanagan.	*/

	if (kindOfSpeaker.compare("Female")) {
		lowerCord.thickness = 1.4e-3;   // dx, in metres
		upperCord.thickness = 0.7e-3;
		cord.length = 10e-3;
		lowerCord.mass = 0.02e-3;   // kilograms
		upperCord.mass = 0.01e-3;
		lowerCord.k1 = 10;   // Newtons per metre
		upperCord.k1 = 4;
	} else if (kindOfSpeaker.compare("Male")) {
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
}

/* End of file Speaker.cpp */
