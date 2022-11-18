
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_output = np.array([[0],[1],[1],[0]])

epochs = 1000
learningRate = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

hidden_weights = np.random.uniform(
    size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(
    size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

print("Initial hidden weights : ")
print(*hidden_weights)
print("Initial biases : ")
print(*hidden_bias)
print("Initial output weights : ")
print(*output_weights)
print("Initial output biases : ")
print(*output_bias)

for i in range(epochs):
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += hidden_bias
    predicted_output = sigmoid(output_layer_activation)

    error = expected_output - predicted_output
    d_predicted_output = error *sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer* sigmoid_derivative(hidden_layer_output)

    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learningRate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims = True) * learningRate
    hidden_weights += inputs.T.dot(d_hidden_layer) * learningRate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims = True) * learningRate

    print("Final hidden weights : ")
    print(hidden_weights)
    print("Final hidden bias : ")
    print(hidden_bias)
    print("Final output weights : ")
    print(output_weights)
    print("Final output bias : ")
    print(output_bias)

    print("\nOutput from neural network after 1000 epochs : ")
    print(*predicted_output)
