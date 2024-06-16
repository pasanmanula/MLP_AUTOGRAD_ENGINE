# MLP Autograd Engine

## Overview

The MLP Autograd Engine is a lightweight C++ library designed for understanding / building and training simple multi-layer perceptrons (MLPs) with automatic differentiation using the first principles of neural networks (Without using any frameworks like Pytorch). This library provides fundamental components to implement a neural network, including scalar values with gradient tracking, neurons, and layers, facilitating backpropagation and gradient descent optimization.

## Features

- **Automatic Differentiation**: Efficiently track and compute gradients for scalar values through the `Value` class.
- **Neural Network Components**: Build neural networks with `Neuron`, `Layer`, and `MLP` classes.
- **Activation Functions**: Apply nonlinear transformations like `tanh` to neural outputs.
- **Gradient-Based Optimization**: Support for gradient propagation and weight updates.

## Components

### Value

The `Value` class represents a scalar with autograd capabilities. It supports basic arithmetic operations and can track gradients through backpropagation.

- **Arithmetic Operations**: Supports `+`, `-`, and `*` operators.
- **Activation Functions**: Includes methods like `tanh` for nonlinear transformations.
- **Backward Propagation**: Computes gradients for backpropagation through the network.

### Neuron

The `Neuron` class models a single neuron. It processes input values using weights and a bias, and applies the `tanh` activation function.

### Layer

The `Layer` class represents a layer of neurons. It aggregates multiple `Neuron` objects to process a set of inputs into outputs.

### MLP

The `MLP` class implements a multi-layer perceptron. It consists of multiple `Layer` objects and supports forward propagation and weight updates through gradient descent.


