import tensorflow as tf
import os


class NeuralNetwork(object):
    def __init__(self, x=None, y=None, train_step=None, **kwargs):
        self.x = x
        self.y = y
        self.train_step = train_step
        # get the quality metrics
        self.metrics = kwargs.get('metrics', [])
        self.train_metrics = kwargs.get('train_metrics', self.metrics)
        self.train_metric_labels = kwargs.get(
            'train_metric_labels', [t.name for t in self.train_metrics]
        )
        self.val_metrics = kwargs.get('val_metrics', self.metrics)
        self.val_metric_labels = kwargs.get(
            'val_metric_labels', [t.name for t in self.val_metrics]
        )
        self.test_metrics = kwargs.get('test_metrics', self.metrics)
        # set up the session
        self.sess = kwargs.get('sess', tf.Session())
        if 'sess' not in kwargs:
            self.sess.run(tf.global_variables_initializer())
        # set up the saver
        self.saver = kwargs.get('saver', tf.train.Saver())
        # set up the logger
        self.auto_logging = kwargs.get('auto_logging', True)
        log_dir = kwargs.get('log_dir', None)
        if log_dir is not None:
            # make directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.train_writer = tf.summary.FileWriter(
                os.path.join(log_dir, 'train'), self.sess.graph
            )
        else:
            self.train_writer = None
        if self.auto_logging:
            for tf_var in tf.global_variables():
                variable_summaries(tf_var)
        self.merged_summaries = kwargs.get('summaries', tf.summary.merge_all())

    @property
    def input(self):
        return self.x

    @property
    def output(self):
        return self.y

    def get_feed_dict(self, dset, n=None, batch_size=None):
        if n is None or batch_size is None:
            x, y = dset.data()
        else:
            x, y = dset.get_batch(n, batch_size)
        return {self.x: x, self.y: y}

    def train(self, d_train, d_val=None, batch_size=128, epochs=1):
        num_batches = d_train.num_batches(batch_size)
        for m in range(epochs):
            print('Epoch {0}/{1}'.format(m + 1, epochs))
            for n in range(num_batches):
                train_feed_dict = self.get_feed_dict(d_train, n, batch_size)

                if n % 100 == 0:
                    train_metrics = self.sess.run( self.train_metrics, feed_dict=train_feed_dict)
                    train_metric_str = ', '.join( ['{0}: {1}'.format(lbl, tm) for tm, lbl in zip(train_metrics, self.train_metric_labels)])
                    print('Batch {0} Training Metrics: {1}'.format( n, train_metric_str))

                    if d_val is not None:
                        val_feed_dict = self.get_feed_dict(d_val)
                        val_metrics = self.sess.run( self.val_metrics, feed_dict=val_feed_dict)
                        val_metric_str = ', '.join( ['{0}: {1}'.format(lbl, vm) for vm, lbl in zip(val_metrics, self.val_metric_labels)])
                        print('Batch {0} Validation Metrics: {1}'.format( n, val_metric_str))

                if self.merged_summaries is not None:
                    merged, _ = self.sess.run( [self.merged_summaries, self.train_step], feed_dict=train_feed_dict)

                    if self.train_writer is not None:
                        self.train_writer.add_summary(merged, m * n)

                else:
                    self.sess.run(self.train_step, feed_dict=train_feed_dict)

    def test(self, d_test):
        test_feed_dict = self.get_feed_dict(d_test)
        return self.sess.run( self.test_metrics, feed_dict=test_feed_dict)

    def predict(self, x):
        return self.sess.run(self.y, feed_dict={self.x: x})

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        self.saver.restore(self.sess, path)

    def close(self):
        self.sess.close()


def variable_summaries(v):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(v)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(v))
        tf.summary.scalar('min', tf.reduce_min(v))
        tf.summary.histogram('histogram', v)
