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

#if _BOOST_
#include <boost/python/numpy.hpp>
#endif

class Artword {
public:
    double totalTime;
    ArtwordData data[kArt_muscle_MAX];
public:
    Artword(double _totalTime);
    Artword(){};
    void Init(double _totalTime);
    void setTarget(int feature, double tim, double value);
    double getTarget(int feature, double tim);
    void removeTarget(int feature, int iTarget);
    void intoArt(Articulation art, double tim);
    void resetTargets();
    void Copy(Artword* newArtword);
#if _BOOST_
    void py_intoArt(boost::python::numpy::ndarray & art, double tim);
#endif
};

/* End of file Artword.h */

#if _BOOST_

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/enum.hpp>
#include <boost/python/class.hpp>

//using namespace boost::python;

BOOST_PYTHON_MODULE(Artword) // tells boost where to look
{

    //boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    //Py_Initialize();
    //boost::python::numpy::initialize();


    boost::python::class_<Artword>("Artword", boost::python::init<double>())
       .def("Init", &Artword::Init)
       .def("setTarget", &Artword::setTarget)
       .def("getTarget", &Artword::getTarget)
       .def("removeTarget", &Artword::removeTarget)
       .def("intoArt", &Artword::py_intoArt)
       .def("resetTargets", &Artword::resetTargets)
    ;

    boost::python::enum_<kArt_muscle>("kArt_muscle") 
        .value("LUNGS",  kArt_muscle_LUNGS)
        .value("INTERARYTENOID", kArt_muscle_INTERARYTENOID)
        .value("CRICOTHYROID", kArt_muscle_CRICOTHYROID)
        .value("VOCALIS",kArt_muscle_VOCALIS   )
        .value("THYROARYTENOID", kArt_muscle_THYROARYTENOID  )
        .value("POSTERIOR_CRICOARYTENOID", kArt_muscle_POSTERIOR_CRICOARYTENOID  )
        .value("LATERAL_CRICOARYTENOID", kArt_muscle_LATERAL_CRICOARYTENOID  )
        .value("STYLOHYOID", kArt_muscle_STYLOHYOID  )
        .value("STERNOHYOID", kArt_muscle_STERNOHYOID  )
        .value("THYROPHARYNGEUS",kArt_muscle_THYROPHARYNGEUS   )
        .value("LOWER_CONSTRICTOR", kArt_muscle_LOWER_CONSTRICTOR  )
        .value("MIDDLE_CONSTRICTOR", kArt_muscle_MIDDLE_CONSTRICTOR  )
        .value("UPPER_CONSTRICTOR", kArt_muscle_UPPER_CONSTRICTOR  )
        .value("SPHINCTER", kArt_muscle_SPHINCTER  )
        .value("HYOGLOSSUS", kArt_muscle_HYOGLOSSUS  )
        .value("STYLOGLOSSUS", kArt_muscle_STYLOGLOSSUS  )
        .value("GENIOGLOSSUS", kArt_muscle_GENIOGLOSSUS  )
        .value("UPPER_TONGUE", kArt_muscle_UPPER_TONGUE  )
        .value("LOWER_TONGUE", kArt_muscle_LOWER_TONGUE  )
        .value("TRANSVERSE_TONGUE", kArt_muscle_TRANSVERSE_TONGUE  )
        .value("VERTICAL_TONGUE", kArt_muscle_VERTICAL_TONGUE  )
        .value("RISORIUS", kArt_muscle_RISORIUS  )
        .value("ORBICULARIS_ORIS", kArt_muscle_ORBICULARIS_ORIS  )
        .value("LEVATOR_PALATINI", kArt_muscle_LEVATOR_PALATINI  )
        .value("TENSOR_PALATINI", kArt_muscle_TENSOR_PALATINI  )
        .value("MASSETER", kArt_muscle_MASSETER  )
        .value("MYLOHYOID", kArt_muscle_MYLOHYOID  )
        .value("LATERAL_PTERYGOID", kArt_muscle_LATERAL_PTERYGOID  )
        .value("BUCCINATOR", kArt_muscle_BUCCINATOR  )
        .value("MIN", kArt_muscle_MIN)
        .value("MAX", kArt_muscle_MAX)
    ;
}
#endif 

#endif
