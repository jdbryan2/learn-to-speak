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
//#include <vector>
//using namespace std;

Delta::Delta (int numberOfTubes) {
    /*
     Preconditions:
        numberOfTubes >= 1;
     Postconditions:
        result -> numberOfTubes = numberOfTubes;
        all members of result -> tube [1..numberOfTubes] are zero or null,
        except 'parallel', which is 1.
     */
    //TODO: Assert (numberOfTubes >= 1);
    this->numberOfTubes = numberOfTubes;
    tube = new vector <structDelta_Tube> (this->numberOfTubes);
    for (int itube = 0; itube < this->numberOfTubes; itube ++) {
        tube->at(itube).parallel = 1;
    }
}
Delta::~Delta () {
	//TODO: Probably need to do something here but I can't remember what needs destroyed
    delete tube;
    /*
    for (int itube = 1; itube <= numberOfTubes; itube ++) {
        delete tube->at(itube);
    }
     */
}

/* End of file Delta.cpp */
