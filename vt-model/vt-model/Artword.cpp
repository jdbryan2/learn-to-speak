/* Artword.cpp
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

#include "Artword.h"
#include <assert.h>
#include <vector>
#include <iostream> // for printing debug, may be removed once bugs are squashed
using namespace std;

/*
	Postconditions:
 my data [feature]. numberOfTargets == 2;
 my data [feature]. times [1] == 0.0;
 my data [feature]. times [2] == self -> totalTime;
 my data [feature]. targets [1] == 0.0;
 my data [feature]. targets [2] == 0.0;
 rest unchanged;
 */

Artword::Artword(double totalTime) {
	this->totalTime = totalTime;
    for (int i = 0; i < kArt_muscle_MAX; i ++) {
        ArtwordData* f = &(data[i]);
        f->numberOfTargets = 2;
        f->targets.push_back(art_target{});
        f->targets.push_back(art_target{totalTime,0.0});
        //f->_iTarget = 1;
    }
}

// TODO: Wow this was bad!!!! I tried to rework some of it, but it needs revisited.
void Artword::setTarget (int feature, double time, double target) {
    assert (feature >= 0);
    assert (feature < kArt_muscle_MAX);
    ArtwordData* f = &(data[feature]);
    assert (f->numberOfTargets >= 2);
    vector<art_target>::iterator it = f->targets.begin();
    // TODO: Not sure this is the desired behavior. May want to revisit.
    // If the desired target time is <0 or Greater than totalTime
    // don't throw an error, just change the time to 0 or totalTime respectively
    if (time < 0.0) {
        time = 0.0;
    }
    else if (time > totalTime) {
        time = totalTime;
        it = f->targets.end();
    }
    else {
        while(time > it->time) {
            it++;
        }
        // Insert element into times and target vectors after insert position.
        if (time != it->time) {
            f->targets.insert(it,{time,target});
            f->numberOfTargets++;
            return;
        }
    }
    *it = {time,target};
}

// TODO: What does this imply about the behavior of the system? Do we want it to interpolate, or do we want it to latch?
// Followup: Do you mean latch as in "zero-order hold" or piece-wise constant? 
//           I think that linear interp is definitely what we want (or some other smooth interp). 
//           The consequence of this is that it won't allow muscles to snap from one target to another - 
//           if they did, the mass-spring system would probably get some nasty oscillations.
//           
// Returns a linear interpolated target at the specified time for the specified feature
double Artword::getTarget (int feature, double time) {
    assert(time <= totalTime && time >= 0.0);
    ArtwordData* f = &(data [feature]);
    vector<art_target>::iterator it = f->targets.begin();
    while(time > it->time) {
        it++;
    }
    // Requested time is a target
    if( it->time == time) {
        return it->target_value;
    }
    // Interpolate y = b + m * x
    else {
        return (it-1)->target_value + (time -(it-1)->time) *
               (it->target_value-(it-1)->target_value) /
               (it->time - (it-1)->time);
    }
}

/*
	Function:
 remove one target from the target list of "feature".
 If "iTarget" is the first or the last target in the list,
 only set the target to zero (begin and end targets remain).
	Preconditions:
 self != nullptr;
 feature in enum Art_MUSCLE;
 iTarget >= 1;
 iTarget <= self -> data [feature]. numberOfTargets;
	Postconditions:
 if (iTarget == 1)
 self -> data [feature]. targets [1] == 0.0;
 else if (iTarget == self -> data [feature]. numberOfTargets)
 self -> data [feature]. targets [iTarget] == 0.0;
 else
 self -> data [feature]. numberOfTargets == old self -> data [feature]. numberOfTargets - 1;
 for (i == iTarget..self -> data [feature]. numberOfTargets)
 self -> data [feature]. times [i] == old self -> data [feature]. times [i + 1];
 self -> data [feature]. targets [i] == old self -> data [feature]. targets [i + 1];
 */


// Note that iTarget is not an index. It is the i'th target so index = iTarget-1;
void Artword::removeTarget (int feature, int iTarget) {
	ArtwordData* f = &(data[feature]);
	assert(iTarget >= 1 || iTarget <= f->numberOfTargets);
	if (iTarget == 1 || iTarget == f->numberOfTargets)
		f -> targets[iTarget-1].target_value = 0.0;
	else {
        f->targets.erase(f->targets.begin()+iTarget-1);
		f -> numberOfTargets --;
	}
}

/*
	Function:
 Linear interpolation between targets, into an existing Art.
	Preconditions:
 me != nullptr;
 art != nullptr;
 */
void Artword::intoArt (Art &art, double time) {
	for (int feature = 0; feature < kArt_muscle_MAX; feature ++) {
		art.art [feature] = getTarget (feature, time);
	}
}

/*
// TODO: Implement something to do graphics for drawing the artword.
void Artword_draw (Artword me, Graphics g, int feature, int garnish) {
	long numberOfTargets = my data [feature]. numberOfTargets;
	if (numberOfTargets > 0) {
		autoNUMvector <double> x (1, numberOfTargets);
		autoNUMvector <double> y (1, numberOfTargets);
		Graphics_setInner (g);
		Graphics_setWindow (g, 0, my totalTime, -1.0, 1.0);
		for (int i = 1; i <= numberOfTargets; i ++) {
			x [i] = my data [feature]. times [i];
			y [i] = my data [feature]. targets [i];
		}
		Graphics_polyline (g, numberOfTargets, & x [1], & y [1]);         
		Graphics_unsetInner (g);
	}

	if (garnish) {
		Graphics_drawInnerBox (g);
		Graphics_marksBottom (g, 2, true, true, false);
		Graphics_marksLeft (g, 3, true, true, true);
		Graphics_textTop (g, false, kArt_muscle_getText (feature));
		Graphics_textBottom (g, true, U"Time (s)");
	}
}
*/

/* End of file Artword.cpp */
