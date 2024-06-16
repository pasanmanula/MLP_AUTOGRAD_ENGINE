#include <gtest/gtest.h>
#include <torch/torch.h>
#include "mlp_engine.hpp"

// Define a neural network class inheriting from torch::nn::Module
class NeuralNet : public torch::nn::Module {
public:
    // Constructor to initialize the network with custom weights
    explicit NeuralNet(
        const std::vector<std::vector<std::vector<std::shared_ptr<Value>>>>& custom_weights)
    {
        // Register and initialize network layers
        input_layer_ = register_module("input_layer", torch::nn::Linear(3, 3));
        hidden_layer_ = register_module("hidden_layer", torch::nn::Linear(3, 2));
        output_layer_ = register_module("output_layer", torch::nn::Linear(2, 1));

        // Ensure weights and biases are in double precision
        convert_weights_to_double();

        // Load custom weights into the network
        initialize_weights(custom_weights);
    }

    // Define the forward pass of the network
    torch::Tensor forward(torch::Tensor x) {
        x = x.to(torch::kDouble);
        x = torch::tanh(input_layer_->forward(x));
        x = torch::tanh(hidden_layer_->forward(x));
        x = torch::tanh(output_layer_->forward(x));
        return x;
    }

    // Utility function to print the gradients of the input layer
    void print_input_layer_gradients() const {
        if (input_layer_->weight.grad().defined()) {
            std::cout << "Input layer weight gradients: " << input_layer_->weight.grad() << std::endl;
        } else {
            std::cout << "No gradients computed for the input layer weights." << std::endl;
        }

        if (input_layer_->bias.grad().defined()) {
            std::cout << "Input layer bias gradients: " << input_layer_->bias.grad() << std::endl;
        } else {
            std::cout << "No gradients computed for the input layer biases." << std::endl;
        }
    }

    // Retrieve the first bias gradient of the input layer
    double retrieve_bias_input_layer() const {
        if (input_layer_->bias.grad().defined()) {
            return input_layer_->bias.grad()[0].item<double>();
        } else {
            std::cout << "No gradients computed for the input layer biases." << std::endl;
            return 0.0;
        }
    }

private:
    // Ensure all weights and biases are in double precision
    void convert_weights_to_double() {
        input_layer_->weight.set_data(input_layer_->weight.to(torch::kDouble));
        input_layer_->bias.set_data(input_layer_->bias.to(torch::kDouble));
        hidden_layer_->weight.set_data(hidden_layer_->weight.to(torch::kDouble));
        hidden_layer_->bias.set_data(hidden_layer_->bias.to(torch::kDouble));
        output_layer_->weight.set_data(output_layer_->weight.to(torch::kDouble));
        output_layer_->bias.set_data(output_layer_->bias.to(torch::kDouble));
    }

    // Initialize network weights with custom provided weights
    void initialize_weights(
        const std::vector<std::vector<std::vector<std::shared_ptr<Value>>>>& custom_weights) {
        if (custom_weights.size() != 3) {
            throw std::runtime_error("Expected weights for 3 layers.");
        }

        set_layer_weights(input_layer_, custom_weights[0]);
        set_layer_weights(hidden_layer_, custom_weights[1]);
        set_layer_weights(output_layer_, custom_weights[2]);
    }

    // Helper function to set weights and biases for a given layer
    void set_layer_weights(
        torch::nn::Linear& layer, const std::vector<std::vector<std::shared_ptr<Value>>>& weights_and_biases) {
        if (weights_and_biases.size() != layer->weight.size(0)) {
            throw std::runtime_error("Mismatch between number of neurons and provided weights.");
        }

        auto weight_tensor = torch::empty({layer->weight.size(0), layer->weight.size(1)}, torch::kDouble);
        auto bias_tensor = torch::empty(layer->bias.size(0), torch::kDouble);

        for (size_t i = 0; i < weights_and_biases.size(); ++i) {
            const auto& neuron_data = weights_and_biases[i];
            if (neuron_data.size() != layer->weight.size(1) + 1) {
                throw std::runtime_error("Mismatch between number of weights and provided weights for neuron.");
            }

            for (size_t j = 0; j < neuron_data.size() - 1; ++j) {
                weight_tensor[i][j] = neuron_data[j]->data_;
            }
            bias_tensor[i] = neuron_data.back()->data_;
        }

        layer->weight.data().copy_(weight_tensor);
        layer->bias.data().copy_(bias_tensor);
    }

    // Define the layers of the network
    torch::nn::Linear input_layer_{nullptr};
    torch::nn::Linear hidden_layer_{nullptr};
    torch::nn::Linear output_layer_{nullptr};
};

// Test suite for core functionalities
class TestCore : public ::testing::Test {};

// Utility function to print gradients for each layer and neuron
void print_all_gradients(const std::vector<std::vector<std::vector<std::shared_ptr<Value>>>>& stored_data) {
    if (stored_data.empty()) {
        std::cout << "The stored data is empty." << std::endl;
        return;
    }

    std::cout << "***********************Begin**************************" << std::endl;
    for (size_t i = 0; i < stored_data.size(); ++i) {
        std::cout << "\nLayer -> " << i << ":" << std::endl;
        if (stored_data[i].empty()) {
            std::cout << "  Empty -> Hint : Ensure record flag is set in MLP" << std::endl;
            continue;
        }

        for (size_t j = 0; j < stored_data[i].size(); ++j) {
            std::cout << "  Neuron -> " << j << ":" << std::endl;
            if (stored_data[i][j].empty()) {
                std::cout << "    Empty -> Hint : Ensure record flag is set in MLP" << std::endl;
                continue;
            }

            for (size_t k = 0; k < stored_data[i][j].size(); ++k) {
                const auto& value_ptr = stored_data[i][j][k];
                if (value_ptr) {
                    if (k != stored_data[i][j].size() - 1) {
                        std::cout << "    Weight - Gradient -> " << k << ": " << value_ptr->gradient_ << std::endl;
                    } else {
                        std::cout << "    Bias - Gradient -> " << k << ": " << value_ptr->gradient_ << std::endl;
                    }
                } else {
                    std::cout << "    Value Obj -> " << k << ": nullptr" << std::endl;
                }
            }
        }
    }
    std::cout << "\n***********************END**************************" << std::endl;
}

// Utility function to print weights for each layer and neuron
void print_all_weights(const std::vector<std::vector<std::vector<std::shared_ptr<Value>>>>& stored_data) {
    if (stored_data.empty()) {
        std::cout << "The stored data is empty." << std::endl;
        return;
    }

    std::cout << "***********************Begin**************************" << std::endl;
    for (size_t i = 0; i < stored_data.size(); ++i) {
        std::cout << "\nLayer -> " << i << ":" << std::endl;
        if (stored_data[i].empty()) {
            std::cout << "  Empty -> Hint : Ensure record flag is set in MLP" << std::endl;
            continue;
        }

        for (size_t j = 0; j < stored_data[i].size(); ++j) {
            std::cout << "  Neuron -> " << j << ":" << std::endl;
            if (stored_data[i][j].empty()) {
                std::cout << "    Empty -> Hint : Ensure record flag is set in MLP" << std::endl;
                continue;
            }

            for (size_t k = 0; k < stored_data[i][j].size(); ++k) {
                const auto& value_ptr = stored_data[i][j][k];
                if (value_ptr) {
                    if (k != stored_data[i][j].size() - 1) {
                        std::cout << "    Weight -> " << k << ": " << value_ptr->data_ << std::endl;
                    } else {
                        std::cout << "    Bias -> " << k << ": " << value_ptr->data_ << std::endl;
                    }
                } else {
                    std::cout << "    Value Obj -> " << k << ": nullptr" << std::endl;
                }
            }
        }
    }
    std::cout << "\n***********************END**************************" << std::endl;
}

// Test case to compare MLP and PyTorch implementations
TEST_F(TestCore, PytorchResultTest) {
    // Define input data for the MLP and PyTorch models
    std::vector<std::shared_ptr<Value>> mlp_input_data =
      {std::make_shared<Value>(1.0), std::make_shared<Value>(2.0), std::make_shared<Value>(3.0) };
    std::vector<double> pytorch_input_data = {1.0, 2.0, 3.0 };
    torch::Tensor pytorch_input =
      torch::tensor(pytorch_input_data, torch::requires_grad().dtype(torch::kDouble)).view({1, 3});

    std::shared_ptr<Value> expected_output = std::make_shared<Value>(1.0);

    // Flag to record weights for testing
    const bool record_weights = true;

    // Instantiate and initialize the MLP
    MLP mlp(3, {3, 2, 1}, record_weights);
    auto mlp_results = mlp(mlp_input_data);
    auto mlp_output = mlp_results.back();

    // Calculate MSE loss for the MLP
    auto loss_diff = mlp_output - expected_output;
    auto mlp_loss = loss_diff->pow(2.0);
    mlp_loss->backward();

    // Retrieve initialized parameters from MLP for comparison
    auto initialized_params = mlp.get_all_parameters();

    // Instantiate the PyTorch network with MLP-initialized weights
    NeuralNet net(initialized_params);
    auto pytorch_output = net.forward(pytorch_input);

    // Compare the output of MLP and PyTorch models
    EXPECT_NEAR(mlp_output->data_, pytorch_output.item<double>(), 0.0001);

    // Calculate and backpropagate PyTorch loss
    torch::nn::MSELoss mse_loss;
    torch::Tensor target = torch::tensor({1.0}, torch::kDouble);
    auto pytorch_loss = mse_loss(pytorch_output, target);
    pytorch_loss.backward();

    // Compare bias gradient of the input layer between MLP and PyTorch models
    EXPECT_NEAR(initialized_params[0][0][0]->gradient_, net.retrieve_bias_input_layer(), 0.0001);

    // Compare the loss value of MLP and PyTorch models
    EXPECT_NEAR(mlp_loss->data_, pytorch_loss.item<double>(), 0.0001);
}
