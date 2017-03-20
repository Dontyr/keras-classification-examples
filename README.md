## Deep Learning Primer with Keras

This repository is a collection of notes on deep learning (i.e., neural networks) with Keras.

### Background

A neural network interface is simple: it multiplies an input variable by a weight and returns a prediction. Weight is like a volume knob. Turning up the weight amplifies a prediction relative to new input.

Neural networks find and create correlation between input data and output data. Finding the best weights that maximize prediction accuracy is a search problem. Getting the best weights requires trying different weights to minimize error, where error is simply the difference between actual output and predicted output. We adjust each weight in the correct direction and correct amount so that the error reduces to zero. "Learning" means adjusting weights to reduce the error to zero.

Here's a simple example of "learning" the best weight by iteratively updating it until the error reaches zero:

``` python
weight = 0.5
prediction_goal = 0.8
network_input = 2.0
alpha = 0.1

for iteration in range(20):
    prediction = network_input * weight
    error = (prediction - prediction_goal) ** 2
    derivative = network_input * (prediction - prediction_goal)
    weight -= alpha * derivative

    print("Iteration:" + str(iteration))
    print("Prediction:" + str(prediction))
    print("Error:" + str(error))
    print("Derivative:" + str(derivative))
    print("Weight:" + str(weight))
    print("*****")
  ```

Adjusting weights to reduce error is really just a search for correlation between input data and output data. If there's no correlation, the error will never reach zero.

### Keras

Keras is a software library that provides high-level building blocks for developing deep learning models. Keras delegates low-level operations such as tensor (i.e. matrix) manipulation and differentiation to a well-optimized tensor library, either Tensorflow or Theanos.

The typical Keras workflow consists of these steps:

* Define training data: input tensors and target tensors.
* Define the network "model", the network of layers that maps inputs to targets.
* Configure the learning process by picking a loss function, an optimizer, and some metrics
to monitor.
* Iterate on the training data.
