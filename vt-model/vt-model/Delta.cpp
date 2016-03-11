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
#include <assert.h>
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
    assert(numberOfTubes >= 0);
    this->numberOfTubes = numberOfTubes;
    // TODO: Don't know if this is a problem, but we are implicitly dynamically allocating memeory for these structs.
    tube.resize(numberOfTubes);
}

// Default Constructor
Delta::Delta () : Delta(89) {}

Delta::~Delta () {
	//TODO: I don't think I need to delete tube because vector's destructor should do it, but I should double check.
}

/* End of file Delta.cpp */
