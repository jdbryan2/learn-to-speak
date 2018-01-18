# Define basic constants for global access
import numpy as np

# define tube sections from PRAAT
TUBES = {}
TUBES['lungs'] = np.append(np.arange(6, 17), 23)
TUBES['full_lungs'] = np.arange(6, 23)
TUBES['bronchi'] = np.arange(23, 29)
TUBES['trachea'] = np.arange(29, 35)
TUBES['glottis'] = np.arange(35, 37)
TUBES['tract'] = np.arange(37, 64)
TUBES['glottis_to_velum'] = np.arange(35, 65)
TUBES['nose'] = np.arange(64, 78)
TUBES['all'] = np.arange(6, 65)  # include velum, exclude nasal cavity
TUBES['all_no_lungs'] = np.arange(23, 65)  # include velum, exclude nasal cavity

# not sure why or if I need this
MAX_NUMBER_OF_TUBES = 89
