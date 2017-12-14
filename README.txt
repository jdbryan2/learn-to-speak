This repository contains
1.)Shell scripts and makefiles for building the Boost-python wrappers for PyRAAT,the python version of the reworked Praat simulator
2.)Python libraries for learning a sensory-motor primitive model of the vocal tract using DFA
3.)Python scripts for using the sensory-motor primitive model to perform control
4.)Python libraries and scripts for performing Q-Learning on the primitve space

The main files that were written for the MDP's and RL project are:
learn-to-speak/python/art_control.py - The script for running the initial primitve control experiments
learn-to-speak/python/learners/Learner.py - The Q-learning class
learn-to-speak/python/art_q_learn.py - The script that performs Q-learning on the primitves

The art_q_learn.py script is a bit messy. This is becuase, as I describe in the report,
I tested out many different state and action space combinations and exploration methods.
I appologize for this. The Learner.py class however is well commented and should be easy
to read through. The art_control.py script is also a bit messy, because it was used for
tuning multiple primitive PID controllers and for generating step responses, but it should
be possible to read through.
Overall, I don't think you will be able to run my code unless you spend a bit of time getting
the Boost PyRAAT libraries working, and are very careful with pointing to the correct distribution
of python. If you really want to run the code though, please take a look at
learn-to-speak/docs/OSX_install.rtf

I really enjoyed this class and am looking forward to testing out other RL approaches on
this problem in the near future.

Thanks,

Jacob Wagner
