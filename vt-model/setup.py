#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="PackageName",
    ext_modules=[
        Extension("vtSim", sources = ["Speaker.cpp", "Sound.cpp", "VocalTract.cpp"],
        libraries = ["boost_python", "portaudio"],
        extra_compile_args=['-std=c++11'])
    ])
