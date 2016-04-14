#ifndef _Articulation_h_
#define _Articulation_h_
/* Articulation.h
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

#include "Articulation_enums.h"

/* Art = Articulation */
/* Members represent muscle activities for speech production. */
/* All members have values from 0 (no activity) to 1 (maximum activity). */

/* Fun fact: In C++, classes are equivalent to structs but are defaulted to 
 *              have private members while structs are defaulted to have 
 *              public members. Otherwise, there isn't any real difference. */
class Art {
public:
    double art[kArt_muscle_MAX]={0}; 
    /* The zero in braces here only defaults the first element to zero manually. 
     * The other elements of this array are set to the default value for doubles,
     * which happens to be zero... */
};

/* End of file Articulation.h */
#endif
