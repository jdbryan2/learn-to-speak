#ifndef _Speaker_def_h_
#define _Speaker_def_h_
/* Speaker_def.h
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


typedef struct  {
    int numberOfMasses;
    double length;
} Speaker_CordDimensions;

typedef struct {
    double thickness;
    double mass;
    double k1;
} Speaker_CordSpring;

typedef struct {
    double Dx;
    double Dy;
    double Dz;
} Speaker_GlottalShunt;

// V
typedef struct {
    double x;
    double y;
    double a;
} Speaker_Velum;

// OM
typedef struct {
    double radius;
} Speaker_Palate;

typedef struct {
    double length;
} Speaker_Tip;

typedef struct {
    double x;
    double y;
    double a;
} Speaker_Alveoli;

typedef struct {
    double dx1;
    double dx2;
    double dy;
} Speaker_TeethCavity;

// rest position of J
typedef struct {
    double r;
    double a;
} Speaker_LowerTeeth;

// U
typedef struct {
    double x;
    double y;
} Speaker_UpperTeeth;

typedef struct {
    double dx;
    double dy;
} Speaker_Lip;

typedef struct {
    double Dx;
    double Dz;
    double weq[14];
} Speaker_Nose;

/* End of file Speaker_def.h */
#endif
