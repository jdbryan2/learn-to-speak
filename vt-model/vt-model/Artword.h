#ifndef _Artword_h_
#define _Artword_h_
/* Artword.h
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
#include "Artword_def.h"

class Artword {
public:
    double totalTime;
    ArtwordData data[kArt_muscle_MAX];
public:
    Artword(double _totalTime);
    void setTarget(int feature, double tim, double value);
    double getTarget(int feature, double tim);
    void removeTarget(int feature, int iTarget);
    void intoArt(double *art, double tim);
    //void Artword_draw (Artword me, Graphics graphics, int feature, int garnish);
};

/* End of file Artword.h */
#endif
