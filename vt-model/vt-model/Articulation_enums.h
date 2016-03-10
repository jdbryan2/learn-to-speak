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
    kArt_muscle_MIN = 0,
    kArt_muscle_ = 0,
    
    kArt_muscle_LUNGS = 1,
    kArt_muscle_INTERARYTENOID = 2, // constriction of larynx; 0 = breathing, 1 = constricted glottis
    
    kArt_muscle_CRICOTHYROID = 3, // vocal-cord tension
    kArt_muscle_VOCALIS = 4, // vocal-cord tension
    kArt_muscle_THYROARYTENOID = 5,
    kArt_muscle_POSTERIOR_CRICOARYTENOID = 6, // opening of glottis
    kArt_muscle_LATERAL_CRICOARYTENOID = 7, // opening of glottis
    
    kArt_muscle_STYLOHYOID = 8, // up movement of hyoid bone
    kArt_muscle_STERNOHYOID = 9, // down movement of hyoid bone
    
    kArt_muscle_THYROPHARYNGEUS = 10, // constriction of ventricular folds
    kArt_muscle_LOWER_CONSTRICTOR = 11,
    kArt_muscle_MIDDLE_CONSTRICTOR = 12,
    kArt_muscle_UPPER_CONSTRICTOR = 13,
    kArt_muscle_SPHINCTER = 14, // constriction of pharynx
    
    kArt_muscle_HYOGLOSSUS = 15, // down movement of tongue body
    kArt_muscle_STYLOGLOSSUS = 16, // up movement of tongue body
    kArt_muscle_GENIOGLOSSUS = 17, // forward movement of tongue body
    
    kArt_muscle_UPPER_TONGUE = 18, // up curling of the tongue tip
    kArt_muscle_LOWER_TONGUE = 19, // down curling of the tongue
    kArt_muscle_TRANSVERSE_TONGUE = 20, // thickening of tongue
    kArt_muscle_VERTICAL_TONGUE = 21, // thinning of tongue
    
    kArt_muscle_RISORIUS = 22, // spreading of lips
    kArt_muscle_ORBICULARIS_ORIS = 23, // rounding of lips
    
    kArt_muscle_LEVATOR_PALATINI = 24, // closing of velo-pharyngeal port; 0 = open ("nasal"), 1 = closed ("oral")
    kArt_muscle_TENSOR_PALATINI = 25,
    
    kArt_muscle_MASSETER = 26,// closing of jaw; 0 = open, 1 = closed
    kArt_muscle_MYLOHYOID = 27, // opening of jaw
    kArt_muscle_LATERAL_PTERYGOID = 28, // horizontal jaw position
    
    kArt_muscle_BUCCINATOR = 29, // oral wall tension
    
    kArt_muscle_DEFAULT = kArt_muscle_LUNGS,
    kArt_muscle_MAX = kArt_muscle_BUCCINATOR
};

/* End of file Articulation.enums */
# endif
