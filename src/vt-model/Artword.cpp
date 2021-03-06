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

#ifndef _BOOST_
#define _BOOST_  1
#endif

#include "Artword.h"
#include <assert.h>
#include <vector>
#include <iostream> // for printing debug, may be removed once bugs are squashed
using namespace std;

Artword::Artword(double _totalTime) {
    Init(_totalTime);
}

void Artword::Init(double _totalTime) {
    this->totalTime = _totalTime;
    for (int i = 0; i < kArt_muscle_MAX; i ++) {
        ArtwordData* f = &(data[i]);
        f->targets.clear();
        vector<art_target>().swap(f->targets);
        f->numberOfTargets = 2;
        f->targets.push_back(art_target{0.0,0.0});
        f->targets.push_back(art_target{totalTime,0.0});
    }
}

void Artword::setTarget (int feature, double time, double target) {
    assert (feature >= 0);
    assert (feature < kArt_muscle_MAX);
    //assert(target>=0);
    //assert(target<=1.0);
    ArtwordData* f = &(data[feature]);
    assert (f->numberOfTargets >= 2);
    vector<art_target>::iterator it = f->targets.begin();
    // If the desired target time is <0 or Greater than totalTime
    // don't throw an error, just change the time to 0 or totalTime respectively
    if (time < 0.0) {
        time = 0.0;
    }
    // Allow entering of a single target value at times greater than totalTime to enable interploation through end of artword
    else if (time > totalTime) {
        // Replace the last target value with this new target
        if (totalTime ==  f->targets.at(f->numberOfTargets-1).time) {
            //f->targets.push_back({time,target});
            f->targets.back() = {time,target};
            f->numberOfTargets++;
            return;
        }
        // If code gets here then you are trying to add more than one articulation beyond totalTime
        assert(0);
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

void Artword::intoArt (Articulation art, double time) {
	for (int feature = 0; feature < kArt_muscle_MAX; feature ++) {
		art [feature] = getTarget (feature, time);
	}
}

void Artword::resetTargets() {
    Init(totalTime);
}

void Artword::Copy(Artword* newArtword) {
    newArtword->totalTime = this->totalTime;
    for (int i = 0; i < kArt_muscle_MAX; i ++) {
        ArtwordData* f_this = &(this->data[i]);
        ArtwordData* f_new = &(newArtword->data[i]);
        f_new->targets.clear();
        for (int j = 0; j < f_this->numberOfTargets; j++) {
            f_new->targets.push_back(f_this->targets[j]);
        }
    }
}

#if _BOOST_
void Artword::py_intoArt(boost::python::numpy::ndarray & art, double tim) {
	for (int feature = 0; feature < kArt_muscle_MAX; feature ++) {
		art [feature] = getTarget (feature, tim);
	}
}
#endif
/* End of file Artword.cpp */
