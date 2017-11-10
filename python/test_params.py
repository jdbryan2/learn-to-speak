
dim = 8
sample_period = 10 # in milliseconds
sample_period=sample_period*8 # convert to samples
#dirname = 'structured_masseter_100'
#dirname = 'full_random_100_primv5'
#dirname = 'full_random_100_primv0'
dirname = 'full_random_20'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
past = 100
future = 10
v_ = 5

# decent results with
# full random 100
# past = 100
# future =10
# sample_period=10
