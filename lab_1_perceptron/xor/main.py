import numpy as np

from xor_model import XorModel


def prepare_data():
    variables = np.array(np.meshgrid([0, 1], [0, 1])).T.reshape(-1, 2)
    xor_result = np.bitwise_xor(variables[:, 0], variables[:, 1])
    return np.array(variables, dtype=np.float32), np.array(xor_result, dtype=np.float32)


def prepare_data_for_4x_xor(split=True):
    variables = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)
    xor_result = np.bitwise_xor(np.bitwise_xor(variables[:, 0], variables[:, 1]),
                                np.bitwise_xor(variables[:, 2], variables[:, 3]))
    indices = np.random.permutation(len(variables))

    variables = variables[indices]
    xor_result = xor_result[indices]

    if not split:
        return np.array(variables, dtype=np.float32), np.array(xor_result, dtype=np.float32)
    return (np.array(variables[:12], dtype=np.float32), np.array(xor_result[:12], dtype=np.float32),
            np.array(variables[12:], dtype=np.float32), np.array(xor_result[12:], dtype=np.float32))


# train_data, train_answers = prepare_data()
flag = True
train_data, train_answers, test_data, test_answers = prepare_data_for_4x_xor()
model = XorModel(4, 16)
model.train(train_data, train_answers, 1000)

if flag:
    predictions = [model.forward(i) for i in train_data]
    train_data_p = train_data.astype(np.int32)
    for i, prediction in enumerate(predictions):
        print(f"{train_data_p[i]} -> {prediction}")
else:
    test_data, test_answers = prepare_data_for_4x_xor(False)
    predictions = [model.forward(i) for i in test_data]
    test_data_p = test_data.astype(np.int32)
    test_answers_p = test_answers.astype(np.int32)
    loss = []
    for i, prediction in enumerate(predictions):
        loss.append(abs(test_answers_p[i] - prediction))
        print(f"{test_data_p[i]} -> {test_answers_p[i]} {prediction}")
    print(f"loss : {sum(loss) / len(loss)}")