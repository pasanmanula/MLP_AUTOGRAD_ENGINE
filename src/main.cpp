#include <iostream>
#include <fstream>
#include "mlp_engine.hpp"

// Custom train method for the mlp engine
void train_mlp(MLP& mlp, 
               const std::vector<std::vector<std::shared_ptr<Value>>>& input_data_batch,
               const std::vector<std::shared_ptr<Value>>& output_ground_truth_batch,
               double step_size, size_t number_of_epochs) {
  for(size_t epoch = 0; epoch < number_of_epochs; epoch++)
  {
    std::vector<std::shared_ptr<Value>> total_loss;
    for(size_t data_index = 0; data_index < input_data_batch.size(); data_index++)
    {
      // Foward Pass
      std::vector<std::shared_ptr<Value>> forward_pass = mlp(input_data_batch.at(data_index));
      std::shared_ptr<Value> output = forward_pass.back();

      // Calculate Individual Loss Per Input Data - MSE
      std::shared_ptr<Value> difference = output - output_ground_truth_batch.at(data_index);
      std::shared_ptr<Value> loss = difference->pow(2.0);

      // Append loss to sum
      total_loss.push_back(loss);
    }

    // Sum up
    auto loss = total_loss.front();
    for(size_t loss_id = 1; loss_id < total_loss.size(); loss_id++)
    {
      loss = loss + total_loss.at(loss_id);
    }

    // Calculate gradient of all the weights and biases (including inputs) in the computational graph
    loss->backward();

    // Update parameters
    mlp.update(step_size);

    std::cout << "Results -> Epoch : " << epoch << " | loss : " << loss->data_ << std::endl;
  }

}

int main()
{

  std::vector<std::vector<std::shared_ptr<Value>>> input_data_batch = {
        {std::make_shared<Value>(1.0), std::make_shared<Value>(2.0), std::make_shared<Value>(3.0)},
        {std::make_shared<Value>(2.0), std::make_shared<Value>(8.0), std::make_shared<Value>(4.0)},
        {std::make_shared<Value>(7.0), std::make_shared<Value>(5.0), std::make_shared<Value>(6.0)},
        {std::make_shared<Value>(4.0), std::make_shared<Value>(2.0), std::make_shared<Value>(9.0)}
    };

  std::vector<std::shared_ptr<Value>> output_ground_truth_batch = {
        std::make_shared<Value>(1.0),
        std::make_shared<Value>(0.75),
        std::make_shared<Value>(0.5),
        std::make_shared<Value>(1.0)
    };

  // Hyperparameters
  const size_t input_dims = input_data_batch.begin()->size();
  const size_t neurons_in_the_first_hidden_layer = 2;
  const size_t neurons_in_the_output_layer = 1;
  const double step_size = 0.001;
  const size_t number_of_epochs = 20;

  // Define A Fully Connected Multi Layer Perceptron
  // {all input + hidden layers + output layer}
  MLP mlp(input_dims,
    {input_dims, neurons_in_the_first_hidden_layer, neurons_in_the_output_layer}, true);

  // Train the MLP
  train_mlp(mlp, input_data_batch, output_ground_truth_batch, step_size, number_of_epochs);
}
