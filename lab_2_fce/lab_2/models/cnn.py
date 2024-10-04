import time

import tensorflow as tf
import numpy as np
from tensorflow.python.trackable.base import Trackable

from models.network import Network
from tools import filer as fw


class CNN(Network, Trackable):

    def __init__(self, hidden_layer_count=1, hidden_neurons=20):
        self._hidden_layer_count = hidden_layer_count
        self._hidden_neurons_count = hidden_neurons
        self._hidden_w_list = []
        self._hidden_b_list = []
        previous_neurons_count = 2
        for i in range(hidden_layer_count):
            self._hidden_w_list.append(
                tf.Variable(tf.random.uniform([previous_neurons_count, hidden_neurons], -1, 1), dtype=tf.float32))
            previous_neurons_count = hidden_neurons
            self._hidden_b_list.append(tf.Variable(tf.zeros([hidden_neurons]), dtype=tf.float32))
        self._output_w = tf.Variable(tf.random.uniform([hidden_neurons * hidden_layer_count + 2, 1], -1, 1),
                                     dtype=tf.float32)
        self._output_b = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        self.deviation = None

        self.checkpoint = tf.train.Checkpoint(
            model=self,
            optimizer=tf.optimizers.Adam()
        )
        self.checkpoint_dir = "checkpoints/cnn_model"
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    def predict(self, x_y):
        _x_y = tf.expand_dims(x_y, 0)
        outputs_list = [_x_y]
        first_hidden_layer = self._activation_relu(tf.matmul(_x_y, self._hidden_w_list[0]) + self._hidden_b_list[0])
        outputs_list.append(first_hidden_layer)
        for i in range(1, self._hidden_layer_count):
            current_hidden_layer = self._activation_relu(
                tf.matmul(tf.concat(outputs_list[1:], axis=0), self._hidden_w_list[i]) + self._hidden_b_list[i])
            outputs_list.append(current_hidden_layer)
        output = tf.matmul(tf.concat(outputs_list, axis=1), self._output_w) + self._output_b
        return output.numpy()[0]

    def _fit_forward(self, batch):
        outputs_list = [batch]
        first_hidden_layer = self._activation_relu(tf.matmul(batch, self._hidden_w_list[0]) + self._hidden_b_list[0])
        outputs_list.append(first_hidden_layer)
        for i in range(1, self._hidden_layer_count):
            current_hidden_layer = self._activation_relu(
                tf.matmul(tf.concat(outputs_list[1:], axis=0), self._hidden_w_list[i]) + self._hidden_b_list[i])
            outputs_list.append(current_hidden_layer)
        output = tf.matmul(tf.concat(outputs_list, axis=1), self._output_w) + self._output_b
        return output

    def _compute_loss(self, output, y):
        return tf.reduce_mean(tf.square(output - y))

    def fit(self, train_data, train_answers, epochs=1000, learning_rate=0.05, batch_size=1):
        print(f"\033[34mCNN:\033[0m\n - hidden layer count: {self._hidden_layer_count}"
              f"\n - hidden neurons count: {self._hidden_neurons_count}\n")
        start_time = time.time()

        self.deviation = max(train_answers)/100
        optimizer = tf.optimizers.SGD(learning_rate)
        loss_list = []
        accuracy_list = []

        for epoch in range(1, epochs + 1):
            epoch_loss = []
            accuracy_numerator = 0

            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answers = train_answers[indexes]

            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_answers = train_answers[i:i + batch_size]
                with (tf.GradientTape() as tape):
                    output = self._fit_forward(batch_data)
                    loss = self._compute_loss(output, batch_answers)
                    epoch_loss.append(loss)

                for _ in range(len(batch_answers)):
                    correct_predictions = np.abs(output.numpy()[_] - batch_answers[_]) < self.deviation
                    accuracy_numerator += np.sum(correct_predictions)

                all_trainable_variables = self._hidden_w_list + self._hidden_b_list + [self._output_w, self._output_b]
                gradients = tape.gradient(loss, all_trainable_variables)
                optimizer.apply_gradients(zip(gradients, all_trainable_variables))

            mean_loss = tf.reduce_mean([l.numpy() for l in epoch_loss])
            loss_list.append(mean_loss.numpy())
            accuracy = accuracy_numerator / len(train_data)
            accuracy_list.append(accuracy)
            if epoch % (epochs // 10) == 0:
                print(f"epoch {epoch:3}/{epochs}, "
                      f"loss={mean_loss:.10f}, "
                      f"accuracy={accuracy}")
        print('\n')
        execution_time = time.time() - start_time
        if sum(accuracy_list[-3:])/3 > 0.9:
            self.save_model()
        statistic = {'network': 'CNN',
                     'accuracy': accuracy_list,
                     'loss': loss_list,
                     'epochs': epochs,
                     'batch_size': batch_size,
                     'execution_time': execution_time,
                     'hidden_layer_count': self._hidden_layer_count,
                     'hidden_neurons_count': self._hidden_neurons_count}
        fw.save_json('cnn_statistic.txt', statistic)
        return statistic

    def _activation_relu(self, x):
        return tf.maximum(0.0, x)

    def save_model(self, checkpoint_dir="checkpoints/cnn_model"):
        save_path = self.checkpoint.save(file_prefix=checkpoint_dir + 'cnn_model')
        print(f"saved to: {save_path}")

    def load_model(self, checkpoint_dir="checkpoints/cnn_model"):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("restored!")
