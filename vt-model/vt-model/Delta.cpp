/* Delta.cpp
 *
 * Copyright (C) 1992-2011,2012,2013,2015,2016 Paul Boersma
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

#include "Delta.h"
#include "support_functions.h"

void Delta::UpdateTube(Speaker &speaker)
{
    double *art = speaker.art;
    double f = speaker.relativeSize * 1e-3;
    double xe [30], ye [30], xi [30], yi [30], xmm [30], ymm [30], dx, dy;
    int closed [40];
    int itube;
    
    // Lungs.
    
    for (itube = 6; itube <= 17; itube ++)
        tube[itube]. Dyeq = 120 * f * (1 + art [kArt_muscle_LUNGS]);
    
    // Glottis.
    
    {
        Delta_Tube t = &(tube[35]);
        t -> Dyeq = f * (5 - 10 * art [kArt_muscle_INTERARYTENOID]
                         + 3 * art [kArt_muscle_POSTERIOR_CRICOARYTENOID]
                         - 3 * art [kArt_muscle_LATERAL_CRICOARYTENOID]);   // 4.38
        t -> k1 = speaker.lowerCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
        t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
    }
    if (speaker.cord.numberOfMasses >= 2) {
        Delta_Tube t = &(tube[36]);
        t -> Dyeq = tube[35]. Dyeq;
        t -> k1 = speaker.upperCord.k1 * (1 + art [kArt_muscle_CRICOTHYROID]);
        t -> k3 = t -> k1 * (20 / t -> Dz) * (20 / t -> Dz);
    }
    if (speaker.cord.numberOfMasses >= 10) {
        tube[83]. Dyeq = 0.75 * 1 * f + 0.25 * tube[35]. Dyeq;
        tube[84]. Dyeq = 0.50 * 1 * f + 0.50 * tube[35]. Dyeq;
        tube[85]. Dyeq = 0.25 * 1 * f + 0.75 * tube[35]. Dyeq;
        tube[83]. k1 = 0.75 * 160 + 0.25 * tube[35]. k1;
        tube[84]. k1 = 0.50 * 160 + 0.50 * tube[35]. k1;
        tube[85]. k1 = 0.25 * 160 + 0.75 * tube[35]. k1;
        for (itube = 83; itube <= 85; itube ++)
            tube[itube]. k3 = tube[itube]. k1 *
            (20 / tube[itube]. Dz) * (20 / tube[itube]. Dz);
    }
    
    // Vocal tract.
    
    Art_Speaker_meshVocalTract (art, speaker, xi, yi, xe, ye, xmm, ymm, closed);
    for (itube = 37; itube <= 63; itube ++) {
        Delta_Tube t = &(tube[itube]);
        // TODO: It appears that he is indexing these other arrays (xmm,ymm,xi,yi,xe,ye,dx,dy) starting @ 1.
        //       So for now let i = itube - 36 so that we start at xmm[37-36=1] instead of xmm[0]
        int i = itube - 36;
        t -> Dxeq = sqrt (( dx = xmm [i] - xmm [i + 1], dx * dx ) + ( dy = ymm [i] - ymm [i + 1], dy * dy ));
        t -> Dyeq = sqrt (( dx = xe [i] - xi [i], dx * dx ) + ( dy = ye [i] - yi [i], dy * dy ));
        if (closed [i]) t -> Dyeq = - t -> Dyeq;
    }
    tube[64]. Dxeq = tube[50]. Dxeq = tube[49]. Dxeq;
    // Voor [r]:  thy tube [59]. Brel = 0.1; thy tube [59]. k1 = 3;
    
    // Nasopharyngeal port.
    
    tube[64]. Dyeq = f * (18 - 25 * art [kArt_muscle_LEVATOR_PALATINI]);   // 4.40
    
    for (itube = 0; itube < numberOfTubes; itube ++) {
        Delta_Tube t = &(tube[itube]);
        t -> s1 = 5e6 * t -> Dxeq * t -> Dzeq;
        t -> s3 = t -> s1 / (0.9e-3 * 0.9e-3);
    }
}

/* End of file Delta.cpp */
