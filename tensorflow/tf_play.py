import tensorflow as tf
import numpy as np

# Create a tensor of shape [2, 3] consisting of random normal values, with mean
# -1 and standard deviation 4.
norm = tf.random_normal([2, 3], mean=-1, stddev=4)

# Shuffle the first dimension of a tensor
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)

features = np.random.random((100, 10))
labels = np.copy(features)

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
output = dataset.batch(5)
print output

output2 = dataset.batch(5)
print output2

## Each time we run these ops, different results are generated
#sess = tf.Session()
#print(sess.run(norm))
#print(sess.run(norm))
#
## Set an op-level seed to generate repeatable sequences across sessions.
#norm = tf.random_normal([2, 3], seed=1234)
#sess = tf.Session()
#print(sess.run(norm))
#print(sess.run(norm))
#sess = tf.Session()
#print(sess.run(norm))
#print(sess.run(norm))
