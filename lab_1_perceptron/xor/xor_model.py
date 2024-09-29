import tensorflow as tf
import numpy as np


def relu(x):
    return tf.maximum(0.0, x)


def tanh(x, derivative=False):
    _tanh = np.tanh(x)
    if derivative:
        return 1 - _tanh ** 2
    return _tanh


def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


class XorModel:
    def __init__(self, input_neurons=2, hidden_neurons=16):
        self.w1 = tf.Variable(tf.random.uniform([input_neurons, hidden_neurons], -1.0, 1.0), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([hidden_neurons]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.uniform([hidden_neurons, 1], -1.0, 1.0), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

        self.checkpoint = tf.train.Checkpoint(w1=self.w1, b1=self.b1, w2=self.  w2, b2=self.b2)

    def forward_train(self, x):
        layer1 = relu(tf.matmul(tf.expand_dims(x, 0), self.w1) + self.b1)
        output = sigmoid(tf.matmul(layer1, self.w2) + self.b2)
        return output

    def forward(self, x):
        layer1 = relu(tf.matmul(tf.expand_dims(x, 0), self.w1) + self.b1)
        output = sigmoid(tf.matmul(layer1, self.w2) + self.b2)
        return round(output.numpy()[0, 0])

    def compute_loss(self, output, y):
        return tf.reduce_mean(tf.square(output - y))

    def train(self, train_data, train_answers, epochs=1000, learning_rate=0.05):
        optimizer = tf.optimizers.SGD(learning_rate)
        loss = None
        for epoch in range(epochs):
            loss_array = []
            for i in range(len(train_data)):
                with tf.GradientTape() as tape:
                    output = self.forward_train(train_data[i])
                    _loss = self.compute_loss(output, train_answers[i])
                    loss_array.append(_loss.numpy())
                gradients = tape.gradient(_loss, [self.w1, self.b1, self.w2, self.b2])
                optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.w2, self.b2]))
            if epoch % 100 == 0:
                loss = sum(loss_array) / len(loss_array)
                print(f"Epoch {epoch}, Loss: {loss}")

        if loss < 0.01:
            self.save_model()

    def save_model(self, checkpoint_dir="checkpoints/"):
        save_path = self.checkpoint.save(file_prefix=checkpoint_dir + 'xor_model')
        print(f"saved to: {save_path}")

    def load_model(self, checkpoint_dir="checkpoints/xor_model"):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("restored!")
