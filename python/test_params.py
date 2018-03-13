
dim = 8
sample_period = 10 # in milliseconds
sample_period=sample_period*8 # # (*8) -> convert to samples ms
dirname = 'data/random'
#dirname = 'full_random_10'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
past = 100
past = 50
future = 10
v_ = 5

# decent results with
# full random 100
# past = 100
# future =10
# sample_period=10
