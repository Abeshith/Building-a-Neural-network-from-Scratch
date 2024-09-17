# Building a Neural Network from Scratch

This project demonstrates how to create a basic neural network from scratch using Python and NumPy. This implementation avoids high-level libraries for layers, optimizers, and activation functions.

## Example Data Used: IRIS

The Iris dataset consists of 150 samples of iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The goal is to classify these flowers into one of three species: setosa, versicolor, or virginica.

## Functions and Explanations

### 1. Activation Functions

#### Sigmoid Function

- **Purpose**: The sigmoid function maps input values to the range (0, 1). It is commonly used in neural networks to introduce non-linearity and to model probabilities in binary classification tasks.
- **Usage**: Applied to the weighted sum of inputs to determine the output of a neuron.

  **Formula**:
 σ(x) = 1 / (1 + e^(-x))

#### Sigmoid Derivative

- **Purpose**: This function computes the gradient of the sigmoid function. The gradient is necessary for the backpropagation algorithm to update weights during training.
- **Usage**: Used to adjust weights during training by calculating how much change in the input affects the output.

  **Formula**:
 σ'(x) = σ(x) * (1 - σ(x))


#### Softmax Function

- **Purpose**: The softmax function converts raw scores (logits) from the output layer into probabilities. It ensures that the sum of probabilities for each sample is 1, making it suitable for multiclass classification.
- **Usage**: Applied to the output layer to obtain class probabilities, allowing the network to predict the class with the highest probability.

  **Formula**:
 softmax(x_i) = e^(x_i) / Σ_j e^(x_j)

### 2. Loss Function

#### Cross-Entropy Loss

- **Purpose**: Cross-entropy loss measures the performance of the classification model by comparing predicted probabilities with the true labels. It penalizes incorrect predictions and helps the model learn by minimizing the difference between predicted and actual values.
- **Usage**: Used to evaluate the model’s performance and guide the training process. The goal is to minimize this loss during training.

  **Formula**:
Loss = - (1 / m) * Σ_i ( Σ_k y_i,k * log(p_i,k) )

  Where:
  - \(m\) is the number of samples,
  - \(K\) is the number of classes,
  - \(y_{i,k}\) is the true label (1 for the correct class, 0 otherwise),
  - \(p_{i,k}\) is the predicted probability for class \(k\).

### 3. Neural Network Components

#### Initialize Weights and Hyperparameters

- **Purpose**: Initializing weights and biases is crucial for training a neural network. Random initialization helps in breaking symmetry and ensuring that neurons learn different features.
- **Usage**: Defines the network's architecture and sets hyperparameters like the learning rate and number of epochs for training.

#### Forward Propagation

- **Purpose**: Forward propagation involves passing the input data through the network layers to compute activations and predictions. It is the process of calculating the output of the network given the current weights and biases.
- **Usage**: Performs the forward pass through the network, computing the activations and output predictions for the training data.

#### Backward Propagation

- **Purpose**: Backward propagation calculates the gradients of the loss function with respect to weights and biases. These gradients are used to update the weights and biases to minimize the loss.
- **Usage**: Performs the backward pass to adjust weights and biases based on the calculated gradients, which helps in improving the model's performance during training.

### 4. Training and Testing

#### Training Loop

- **Purpose**: The training loop iteratively performs forward and backward propagation, updates weights, and monitors the loss. It helps the model learn from the training data by adjusting weights based on the computed gradients.
- **Usage**: Runs for a specified number of epochs to fit the model to the training data, with regular updates to the weights and biases to minimize the loss.

#### Testing the Model

- **Purpose**: Testing evaluates the model's performance on unseen data (test set). It calculates accuracy by comparing the predicted class labels with the true labels to assess how well the model generalizes.
- **Usage**: Provides a measure of how effectively the trained model can classify new data, indicating the model's overall performance.

## Conclusion

This implementation of a neural network from scratch demonstrates fundamental concepts and techniques used in machine learning. By manually coding the activation functions, loss function, and training processes, you gain a deeper understanding of how neural networks operate and learn. This approach provides valuable insights into the mechanics of neural networks and lays the groundwork for more advanced machine learning projects.

