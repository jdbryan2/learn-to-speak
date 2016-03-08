#ifndef _Speaker_h_
#define _Speaker_h_
/* Speaker.h
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

#include "Speaker_def.h"
#include <string>
using namespace std;

class Speaker {
//public:
    double relativeSize;  // different for female, male, child
    /* In the larynx. */
    Speaker_CordDimensions cord;
    Speaker_CordSpring lowerCord;
    Speaker_CordSpring upperCord;
    Speaker_GlottalShunt shunt;
    
    /* Above the larynx. */
    
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
    
    /* In the nasal cavity. */
    
    Speaker_Nose nose;
    
public:
    Speaker(string, int);
};

/* End of file Speaker.h */
#endif
