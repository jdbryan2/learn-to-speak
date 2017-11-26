
dim = 1
sample_period_ms = 20 # in milliseconds
sample_period=sample_period_ms*8 # convert to samples
#dirname = 'structured_masseter_100'
#dirname = 'full_random_100_primv5'
#dirname = 'full_random_100_primv0'
dirname = 'full_random_100_qlearn'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
past = 12
future = 12
v_ = 5

# decent results with
# full random 100
# past = 100
# future =10
# sample_period=10
