/*
 * ------------------------------------------------------------------------
 * 
 *  Project:        MLP Autograd Engine
 *  Filename:       mlp_engine.hpp
 *  Author:         Pasan Bandara
 *  
 *  Description:    
 *  This header file defines the core components of a multi-layer perceptron (MLP)
 *  with automatic differentiation capabilities. The main classes and functions 
 *  provided include:
 *  
 *  - **Value**: A class representing a scalar value that supports basic arithmetic 
 *    operations and tracks gradients for automatic differentiation. It includes 
 *    methods for applying activation functions and performing backpropagation.
 *  
 *  - **Neuron**: A class representing a single neuron in a neural network, which 
 *    processes inputs and applies a tanh activation function.
 *  
 *  - **Layer**: A class representing a layer of neurons, which computes the outputs 
 *    for a given set of inputs by propagating them through its neurons.
 *  
 *  - **MLP**: A class representing a multi-layer perceptron, which is a neural 
 *    network composed of multiple layers. It supports forward propagation of inputs 
 *    and updating of weights based on computed gradients.
 *  
 *  - **Operator Overloads**: Overloaded operators for addition, subtraction, and 
 *    multiplication of shared pointers to `Value` objects, enabling these operations 
 *    to support gradient tracking.
 * 
 *  This file facilitates the construction and training of neural networks with 
 *  gradient-based optimization. It is intended for educational purposes and provides 
 *  a foundational understanding of backpropagation and gradient descent.
 */

#ifndef MLP_AUTOGRAD_ENGINE_MLP_ENGINE_HPP_
#define MLP_AUTOGRAD_ENGINE_MLP_ENGINE_HPP_

#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <set>
#include <random>
#include <algorithm>

const size_t INPUT_LAYER_ID = 0; // Constant for the input layer ID

// Forward declaration of Value class to allow friend function declarations
class Value;

// Operator overloading for shared_ptr<Value> addition
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);

// Operator overloading for shared_ptr<Value> multiplication
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);

// Class representing a scalar value with autograd capabilities
class Value : public std::enable_shared_from_this<Value> {
public:
    explicit Value(double data, std::shared_ptr<Value> child1 = nullptr,
                   std::shared_ptr<Value> child2 = nullptr, std::string operation = "")
        : data_(data), gradient_(0.0), child1_(std::move(child1)),
          child2_(std::move(child2)), operation_(std::move(operation)) {}

    // Apply tanh activation and return the resulting Value
    std::shared_ptr<Value> tanh() {
        double x = data_;
        double tanh_value = std::tanh(x);
        auto shared_this = shared_from_this();
        auto output = std::make_shared<Value>(tanh_value, shared_this, nullptr, "tanh");

        std::weak_ptr<Value> weak_this = shared_this;
        output->backward_function_ = [weak_this, tanh_value, output]() {
            if (auto shared_this = weak_this.lock()) {
                shared_this->gradient_ += (1 - tanh_value * tanh_value) * output->gradient_;
            } else {
                std::cerr << "Warning: the object has been destroyed. Operation: tanh" << std::endl;
            }
        };

        return output;
    }

    // Apply power operation and return the resulting Value
    std::shared_ptr<Value> pow(double exponent) {
        double pow_value = std::pow(data_, exponent);
        auto shared_this = shared_from_this();
        auto output = std::make_shared<Value>(pow_value, shared_this, nullptr, "pow");

        std::weak_ptr<Value> weak_this = shared_this;
        output->backward_function_ = [weak_this, exponent, output]() {
            if (auto shared_this = weak_this.lock()) {
                shared_this->gradient_ +=
                  (exponent * std::pow(shared_this->data_, exponent - 1.0)) * output->gradient_;
            } else {
                std::cerr << "Warning: the object has been destroyed. Operation: pow" << std::endl;
            }
        };

        return output;
    }

    // Perform backward propagation to calculate gradients
    void backward() {
        std::function<void(std::shared_ptr<Value>)> build_topological_order = [&](std::shared_ptr<Value> node) {
            if (!node || visited_.count(node)) return;
            visited_.insert(node);
            if (node->child1_) build_topological_order(node->child1_);
            if (node->child2_) build_topological_order(node->child2_);
            topological_order_.emplace_back(node);
        };

        gradient_ = 1.0;  // Initialize gradient at the root to 1.0
        topological_order_.clear();
        visited_.clear();

        auto self = shared_from_this();
        build_topological_order(self);
        std::reverse(topological_order_.begin(), topological_order_.end());

        for (const auto& node : topological_order_) {
            node->execute_backward_function();
        }
    }

    // Print function for Value object
    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& value) {
        os << "{Value -> data: " << value->data_ << ", gradient: " << value->gradient_;
        if (value->operation_.empty()) {
            os << " | Leaf node}";
        } else {
            os << " | Operation: " << value->operation_ << "}";
        }
        return os;
    }

    double data_;
    double gradient_;
    std::shared_ptr<Value> child1_;
    std::shared_ptr<Value> child2_;
    std::string operation_;
    std::function<void()> backward_function_;
    std::vector<std::shared_ptr<Value>> topological_order_;
    std::set<std::shared_ptr<Value>> visited_;

private:
    void execute_backward_function() {
        if (backward_function_) {
            backward_function_();
        }
    }
};

// Neuron class representing a single neuron in the network
class Neuron {
public:
    explicit Neuron(size_t input_dim, bool record_weights = false)
        : input_dim_(input_dim) {
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (size_t i = 0; i < input_dim_; ++i) {
            double weight_value = dis(gen);
            weights_.emplace_back(std::make_shared<Value>(weight_value));
        }

        double bias_value = dis(gen);
        bias_ = std::make_shared<Value>(bias_value);

        if (record_weights) {
            recorded_params_ = weights_;
            recorded_params_.push_back(bias_);
        }
    }

    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        if (inputs.size() != weights_.size()) {
            std::cerr << "Error: Mismatched input and weight dimensions." << std::endl;
            return std::make_shared<Value>(0.0);
        }

        auto sum = bias_;
        for (size_t i = 0; i < inputs.size(); ++i) {
            sum = sum + (inputs[i] * weights_[i]);
        }

        return sum->tanh();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Value>> get_parameters() const {
        return recorded_params_;
    }

private:
    size_t input_dim_;
    std::vector<std::shared_ptr<Value>> weights_;
    std::shared_ptr<Value> bias_;
    std::vector<std::shared_ptr<Value>> recorded_params_;
};

// Layer class representing a layer of neurons
class Layer {
public:
    Layer(size_t input_dim, size_t num_neurons, bool record_weights = false)
        : input_dim_(input_dim), num_neurons_(num_neurons) {
        for (size_t i = 0; i < num_neurons_; ++i) {
            neurons_.emplace_back(Neuron(input_dim_, record_weights));
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
        std::vector<std::shared_ptr<Value>> outputs;
        for (auto& neuron : neurons_) {
            outputs.emplace_back(neuron(inputs));
        }
        return outputs;
    }

    [[nodiscard]] std::vector<std::vector<std::shared_ptr<Value>>> get_parameters() const {
        std::vector<std::vector<std::shared_ptr<Value>>> all_params;
        for (const auto& neuron : neurons_) {
            all_params.emplace_back(neuron.get_parameters());
        }
        return all_params;
    }

private:
    size_t input_dim_;
    size_t num_neurons_;
    std::vector<Neuron> neurons_;
};

// Multi-layer perceptron class representing the neural network
class MLP {
public:
    MLP(size_t input_dim, const std::vector<size_t>& layers, bool record_weights = false)
        : input_dim_(input_dim) {
        if (layers.empty()) {
            throw std::invalid_argument("The network must have at least one layer.");
        }

        layers_.emplace_back(Layer(input_dim_, layers[INPUT_LAYER_ID], record_weights));
        for (size_t i = 1; i < layers.size(); ++i) {
            layers_.emplace_back(Layer(layers[i - 1], layers[i], record_weights));
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> inputs) {
        for (auto& layer : layers_) {
            inputs = layer(inputs);
        }
        return inputs;
    }

    [[nodiscard]] std::vector<std::vector<std::vector<std::shared_ptr<Value>>>> get_all_parameters() const {
        std::vector<std::vector<std::vector<std::shared_ptr<Value>>>> all_params;
        for (const auto& layer : layers_) {
            all_params.emplace_back(layer.get_parameters());
        }
        return all_params;
    }

    void update(double step_size) {
        auto all_params = get_all_parameters();
        for (auto& layer_params : all_params) {
            for (auto& neuron_params : layer_params) {
                for (auto& param : neuron_params) {
                    param->data_ -= step_size * param->gradient_;
                }
            }
        }
    }

private:
    size_t input_dim_;
    std::vector<Layer> layers_;
};

// Free function for addition of shared_ptr<Value>
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto output = std::make_shared<Value>(a->data_ + b->data_, a, b, "+");

    output->backward_function_ = [a, b, output]() {
        a->gradient_ += output->gradient_;
        b->gradient_ += output->gradient_;
    };

    return output;
}

// Free function for subtraction of shared_ptr<Value>
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto output = std::make_shared<Value>(a->data_ - b->data_, a, b, "-");

    output->backward_function_ = [a, b, output]() {
        a->gradient_ += output->gradient_;
        b->gradient_ += output->gradient_;
    };

    return output;
}

// Free function for multiplication of shared_ptr<Value>
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    auto output = std::make_shared<Value>(a->data_ * b->data_, a, b, "*");

    output->backward_function_ = [a, b, output]() {
        a->gradient_ += b->data_ * output->gradient_;
        b->gradient_ += a->data_ * output->gradient_;
    };

    return output;
}

#endif // MLP_AUTOGRAD_ENGINE_MLP_ENGINE_HPP_
