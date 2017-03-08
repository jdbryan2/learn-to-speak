#ifndef _Articulation_enums_h_
#define _Articulation_enums_h_
/* Articulation_enums.h
 *
 * Copyright (C) 1992-2009,2015 Paul Boersma
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

enum kArt_muscle {
    kArt_muscle_LUNGS = 0,
    kArt_muscle_INTERARYTENOID = 1, // constriction of larynx; 0 = breathing, 1 = constricted glottis
    
    kArt_muscle_CRICOTHYROID = 2, // vocal-cord tension
    kArt_muscle_VOCALIS = 3, // vocal-cord tension
    kArt_muscle_THYROARYTENOID = 4,
    kArt_muscle_POSTERIOR_CRICOARYTENOID = 5, // opening of glottis
    kArt_muscle_LATERAL_CRICOARYTENOID = 6, // opening of glottis
    
    kArt_muscle_STYLOHYOID = 7, // up movement of hyoid bone
    kArt_muscle_STERNOHYOID = 8, // down movement of hyoid bone
    
    kArt_muscle_THYROPHARYNGEUS = 9, // constriction of ventricular folds
    kArt_muscle_LOWER_CONSTRICTOR = 10,
    kArt_muscle_MIDDLE_CONSTRICTOR = 11,
    kArt_muscle_UPPER_CONSTRICTOR = 12,
    kArt_muscle_SPHINCTER = 13, // constriction of pharynx
    
    kArt_muscle_HYOGLOSSUS = 14, // down movement of tongue body
    kArt_muscle_STYLOGLOSSUS = 15, // up movement of tongue body
    kArt_muscle_GENIOGLOSSUS = 16, // forward movement of tongue body
    
    kArt_muscle_UPPER_TONGUE = 17, // up curling of the tongue tip
    kArt_muscle_LOWER_TONGUE = 18, // down curling of the tongue
    kArt_muscle_TRANSVERSE_TONGUE = 19, // thickening of tongue
    kArt_muscle_VERTICAL_TONGUE = 20, // thinning of tongue
    
    kArt_muscle_RISORIUS = 21, // spreading of lips
    kArt_muscle_ORBICULARIS_ORIS = 22, // rounding of lips
    
    kArt_muscle_LEVATOR_PALATINI = 23, // closing of velo-pharyngeal port; 0 = open ("nasal"), 1 = closed ("oral")
    kArt_muscle_TENSOR_PALATINI = 24,
    
    kArt_muscle_MASSETER = 25,// closing of jaw; 0 = open, 1 = closed
    kArt_muscle_MYLOHYOID = 26, // opening of jaw
    kArt_muscle_LATERAL_PTERYGOID = 27, // horizontal jaw position
    
    kArt_muscle_BUCCINATOR = 28, // oral wall tension
    
    kArt_muscle_DEFAULT = kArt_muscle_LUNGS,
    kArt_muscle_MIN = kArt_muscle_LUNGS,
    kArt_muscle_MAX = kArt_muscle_BUCCINATOR+1 // +1 so that this can be used to initialize arrays properly
};

// TODO: Probably need to make this into its own class. And add a copy function to overload =
typedef double Articulation[kArt_muscle_MAX];

/* End of file Articulation.enums */
# endif
