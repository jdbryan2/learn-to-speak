from neural_network import NeuralNetwork


class Autoencoder(NeuralNetwork):
    def __init__(self, encoder=None, decoder=None, train_step=None, **kwargs):
        if not isinstance(encoder, NeuralNetwork):
            raise ValueError('Encoder must be a neural network')
        if not isinstance(decoder, NeuralNetwork):
            raise ValueError('Decoder must be a neural network')
        NeuralNetwork.__init__(self, encoder.input, decoder.output, train_step, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.sess.run(self.encoder.output, {self.encoder.input: x})

    def decode(self, y):
        return self.sess.run(self.decoder.output, {self.decoder.input: y})

    # note: self.target must get defined in the inheriting class
    def get_feed_dict(self, dset, n=None, batch_size=None):
        if n is None or batch_size is None:
            x, target = dset.data()
        else:
            x, target = dset.get_batch(n, batch_size)
        return {self.x: x, self.target: target}

    # Function removed so that encoder can be learned over an input-output space
    # traditional autoencoder requires that data fed by the get_batch function
    # returns two copies of the same data.
    ############################################################################
    #def get_feed_dict(self, dset, n=None, batch_size=None):
    #    if n is None or batch_size is None:
    #        x = dset.data()
    #    else:
    #        x = dset.get_batch(n, batch_size)
    #    return {self.x: x}

