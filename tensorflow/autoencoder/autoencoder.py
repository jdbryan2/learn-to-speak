from neural_network import NeuralNetwork


class Autoencoder(NeuralNetwork):
    def __init__(self, encoder=None, decoder=None, train_step=None, **kwargs):
        if not isinstance(encoder, NeuralNetwork):
            raise ValueError('Encoder must be a neural network')
        if not isinstance(decoder, NeuralNetwork):
            raise ValueError('Decoder must be a neural network')
        NeuralNetwork.__init__(
            self, encoder.input, decoder.output, train_step, **kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def get_feed_dict(self, dset, n=None, batch_size=None):
        if n is None or batch_size is None:
            x = dset.data()
        else:
            x = dset.get_batch(n, batch_size)
        return {self.x: x}

    def encode(self, x):
        return self.sess.run(self.encoder.output, {self.encoder.input: x})

    def decode(self, y):
        return self.sess.run(self.decoder.output, {self.decoder.input: y})
