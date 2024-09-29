import tensorflow as tf
import numpy as np


def prepare_data():
    variables = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)

    xor_result = np.bitwise_xor(np.bitwise_xor(variables[:, 0], variables[:, 1]),
                                np.bitwise_xor(variables[:, 2], variables[:, 3]))

    return np.array(variables), np.array(xor_result)


train_data, train_answers = prepare_data()
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_answers, epochs=100)

loss, accuracy = model.evaluate(train_data, train_answers)
print("loss", loss)
print("accuracy", accuracy)

prediction = model.predict(train_data)
for inp, pred in zip(train_data, prediction):
    print(inp, round(pred[0]))
