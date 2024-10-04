import json
import os


def save_json(file_name, statistic):
    statistic['accuracy'] = [float(i) for i in statistic['accuracy']]
    statistic['loss'] = [float(i) for i in statistic['loss']]
    with open("data_files/" + file_name, 'a') as file:
        file.write(json.dumps(statistic) + '\n')


def save(file_path, data):
    with open(file_path, 'a') as file:
        file.write(data + '\n')


def read(file_name):
    with open("data_files/" + file_name, 'r') as file:
        data = file.read()
    return data


def read_from_dir(directory="data_files/"):
    data = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            tmp = []
            with open(file_path, 'r') as f:
                for line in f:
                    tmp.append(json.loads(line))
            data[filename] = tmp
    return data
