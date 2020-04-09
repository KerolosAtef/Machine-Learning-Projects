import tensorflow as tf
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0], dtype=float)
    ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 550.0], dtype=float)
    model = keras.models.Sequential([keras.layers.Dense(units=1, input_shape=[1])])  # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1)
    return model.predict(y_new)[0]


@app.route('/')
def hello():
    return "kirolos atef"


@app.route('/print_input', methods=['GET'])
def mmm():
    return jsonify({"output": request.get_json('name')['name'], })


@app.route('/output', methods=['POST'])
def jason_output():
    prediction = house_model([7.0])
    return {"predicted value": str(prediction)}


if __name__ == '__main__':
    app.run(debug=True)
