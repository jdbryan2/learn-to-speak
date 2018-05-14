
dim = 8
#sample_period_ms = 20 # in milliseconds
sample_period_ms = 1 # in milliseconds
sample_period=sample_period_ms*8 # # (*8) -> convert to samples ms
#dirname = 'data/batch_random_12_12'
dirname = 'data/batch_random_20_5'
#dirname = 'data/batch_random_20_10'
#dirname = 'full_random_10'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path
load_fname = 'round257.npz' # class points toward 'data/' already, just need the rest of the path
#past = 50
#future = 10
past = 20 
future = 5
#v_ = 5

# decent results with
# full random 100
# past = 100
# future =10
# sample_period=10
