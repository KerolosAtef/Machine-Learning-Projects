import csv
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
import numpy as np


def koko_train():
    sum_age_of_lived = 0
    num_of_lived = 0
    num_of_died = 0
    sum_age_of_died = 0
    with open("train.csv", "r") as fread:
        data = csv.reader(fread, delimiter=',')
        with open("koko_train.csv", "w") as w:
            for row in data:
                i = -1
                survived = ""
                for k in row:
                    i += 1
                    if i == 1:
                        survived = k
                    if i == 5 and survived == "1":
                        # num_of_lived += 1
                        k = "24" if k == '' else k
                        # sum_age_of_lived += float(k) if k != '' else 0
                    elif i == 5 and survived == "0":
                        # num_of_died += 1
                        k = "23.6" if k == '' else k
                        # sum_age_of_died += float(k) if k != '' else 0
                    if i == 0 or i == 3 or i == 8 or i == 10:
                        continue
                    elif i == 2 and k == "":
                        k = "0,0,0"
                    elif i == 2 and k == "1":
                        k = "1,0,0"
                    elif i == 2 and k == "2":
                        k = "0,1,0"
                    elif i == 2 and k == "3":
                        k = "0,0,1"
                    elif i == 4 and k == "male":
                        k = "1"
                    elif i == 4 and k == "female":
                        k = "0"
                    elif i == 11 and k == "":
                        k = "0,0,0"
                    elif i == 11 and k == "S":
                        k = "1,0,0"
                    elif i == 11 and k == "C":
                        k = "0,1,0"
                    elif i == 11 and k == "Q":
                        k = "0,0,1"

                    if i == 11:
                        w.write(k)
                    else:
                        w.write(k + ",")
                w.write("\n")
    # print(sum_age_of_died/num_of_died)
    # print(sum_age_of_lived/num_of_lived)


def koko_test():
    with open("test.csv", "r") as fread:
        data = csv.reader(fread, delimiter=',')
        with open("koko_test.csv", "w") as w:
            for row in data:
                i = -1
                for k in row:
                    i += 1
                    if i == 0 or i == 2 or i == 7 or i == 9:
                        continue
                    elif i == 1 and k == "":
                        k = "0,0,0"
                    elif i == 1 and k == "1":
                        k = "1,0,0"
                    elif i == 1 and k == "2":
                        k = "0,1,0"
                    elif i == 1 and k == "3":
                        k = "0,0,1"
                    elif i == 3 and k == "male":
                        k = "1"
                    elif i == 3 and k == "female":
                        k = "0"
                    elif i == 4 and k == '':
                        k = "24"
                    elif i == 10 and k == "":
                        k = "0,0,0"
                    elif i == 10 and k == "S":
                        k = "1,0,0"
                    elif i == 10 and k == "C":
                        k = "0,1,0"
                    elif i == 10 and k == "Q":
                        k = "0,0,1"

                    if i == 10:
                        w.write(k)
                    else:
                        w.write(k + ",")
                w.write("\n")


def read_train_dataset():
    x = []
    y = []
    with open("koko_train.csv", "r") as file:
        data = csv.reader(file, delimiter=',')
        next(data)
        for row in data:
            i = -1
            temp_x = []
            for k in row:
                i = i + 1
                if i == 0:
                    y.append(k)
                else:
                    temp_x.append(float(k))
            x.append(temp_x)

    return x, y


def read_test_dataset():
    x = []
    with open("koko_test.csv", "r") as file:
        data = csv.reader(file, delimiter=',')
        next(data)
        for row in data:
            temp_x = []
            for k in row:
                temp_x.append(float(k))
            x.append(temp_x)

    return x


def split_data(x, y):
    return x[:int(0.8 * len(x))], x[int(0.8 * len(x)):], y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]


if __name__ == '__main__':
    # koko_train()
    # koko_test()
    x, y = read_train_dataset()
    x_train, x_validate, y_train, y_validate = split_data(x, y)
    x_test = read_test_dataset()
    x = np.array(x, dtype='float')
    y = np.array(y, dtype='int')
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='int')
    x_validate = np.array(x_validate, dtype='float')
    y_validate = np.array(y_validate, dtype='int')
    x_test = np.array(x_test, dtype='float')
    print(x)
    print(y)
    model = tf.keras.models.Sequential(layers=[
        Dense(units=512, activation='relu', input_shape=[11]),
        # Dense(units=32, activation='relu'),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=500, validation_data=(x_validate, y_validate))
    model.evaluate(x_validate, y_validate)
    model.save("koko_model.h5")
    print("model saved")
    # model = tf.keras.models.load_model("koko_model.h5")
    # prediction
    # print(x_test[0])
    with open("output.csv", 'w') as fout:
        fout.write("PassengerId,Survived\n")
        count = 892
        output = model.predict(x_test)
        for p in output:
            fout.write(str(count) + "," + str(1 if p[0] > 0.5 else 0) + "\n")
            count += 1
