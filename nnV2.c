// Include necessary header files
#include <stdio.h>  // For input/output operations (printf, scanf, etc.)
#include <stdlib.h> // For rand() function and dynamic memory allocation
#include <math.h>   // For mathematical functions like exp()

// Define the structure of the neural network
#define INPUT_NEURONS 4     // Number of input neurons (4 for 4-bit input)
#define HIDDEN_NEURONS 8    // Number of neurons in the hidden layer
#define OUTPUT_NEURONS 1    // Number of output neurons (1 for binary classification)
#define LEARNING_RATE 0.1   // Step size for weight updates
#define NUM_TRAINING_EXAMPLES 16 // Total number of training examples (2^4 for 4-bit input)

// Activation function: Rectified Linear Unit (ReLU)
// ReLU returns the input if it's positive, otherwise it returns 0
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Derivative of ReLU function
// Used in backpropagation to calculate gradients
double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

// Activation function: Sigmoid
// Maps any input to a value between 0 and 1
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of sigmoid function
// Used in backpropagation to calculate gradients for output layer
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Initialize weights with random values between -1 and 1
void init_weights(double weights[], int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

// Function to generate training data for 4-bit input
// Creates all 16 possible 4-bit combinations and their corresponding outputs
void generate_training_data(double inputs[NUM_TRAINING_EXAMPLES][INPUT_NEURONS], 
                            double outputs[NUM_TRAINING_EXAMPLES][OUTPUT_NEURONS]) {

    // DESIRED OUTPUT FUNCTION.
    int output_pattern[16] = {0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0};
    
    for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
        // Generate 4-bit binary input (counting up from 0 to 15)
        for (int j = 0; j < INPUT_NEURONS; j++) {
            inputs[i][j] = (i >> j) & 1;
        }
        // Assign the corresponding output from the predefined pattern
        outputs[i][0] = output_pattern[i];
    }
}

int main() {
    // Declare weight matrices and bias vectors
    double hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS];
    double output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS];
    double hidden_bias[HIDDEN_NEURONS];
    double output_bias[OUTPUT_NEURONS];

    // Initialize weights and biases with random values
    init_weights(hidden_weights, INPUT_NEURONS * HIDDEN_NEURONS);
    init_weights(output_weights, HIDDEN_NEURONS * OUTPUT_NEURONS);
    init_weights(hidden_bias, HIDDEN_NEURONS);
    init_weights(output_bias, OUTPUT_NEURONS);

    // Generate training data for 4-bit input
    double training_inputs[NUM_TRAINING_EXAMPLES][INPUT_NEURONS];
    double training_outputs[NUM_TRAINING_EXAMPLES][OUTPUT_NEURONS];
    generate_training_data(training_inputs, training_outputs);

    // Training loop
    for (int epoch = 0; epoch < 10000; epoch++) {
        // Iterate through each training example
        for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
            // Forward propagation

            // Calculate hidden layer activations
            double hidden_layer[HIDDEN_NEURONS];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                double sum = hidden_bias[j];
                for (int k = 0; k < INPUT_NEURONS; k++) {
                    sum += training_inputs[i][k] * hidden_weights[k * HIDDEN_NEURONS + j];
                }
                hidden_layer[j] = relu(sum);
            }

            // Calculate output layer activations
            double output_layer[OUTPUT_NEURONS];
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                double sum = output_bias[j];
                for (int k = 0; k < HIDDEN_NEURONS; k++) {
                    sum += hidden_layer[k] * output_weights[k * OUTPUT_NEURONS + j];
                }
                output_layer[j] = sigmoid(sum);
            }

            // Backpropagation

            // Calculate output layer error
            double output_error[OUTPUT_NEURONS];
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                output_error[j] = (training_outputs[i][j] - output_layer[j]) * sigmoid_derivative(output_layer[j]);
            }

            // Calculate hidden layer error
            double hidden_error[HIDDEN_NEURONS];
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                hidden_error[j] = 0;
                for (int k = 0; k < OUTPUT_NEURONS; k++) {
                    hidden_error[j] += output_error[k] * output_weights[j * OUTPUT_NEURONS + k];
                }
                hidden_error[j] *= relu_derivative(hidden_layer[j]);
            }

            // Update weights and biases for the output layer
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                output_bias[j] += LEARNING_RATE * output_error[j];
                for (int k = 0; k < HIDDEN_NEURONS; k++) {
                    output_weights[k * OUTPUT_NEURONS + j] += LEARNING_RATE * output_error[j] * hidden_layer[k];
                }
            }

            // Update weights and biases for the hidden layer
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                hidden_bias[j] += LEARNING_RATE * hidden_error[j];
                for (int k = 0; k < INPUT_NEURONS; k++) {
                    hidden_weights[k * HIDDEN_NEURONS + j] += LEARNING_RATE * hidden_error[j] * training_inputs[i][k];
                }
            }
        }
    }

    // Test the trained network
    printf("Testing the neural network:\n");
    int correct = 0;
    for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
        // Forward propagation for testing
        double hidden_layer[HIDDEN_NEURONS];
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            double sum = hidden_bias[j];
            for (int k = 0; k < INPUT_NEURONS; k++) {
                sum += training_inputs[i][k] * hidden_weights[k * HIDDEN_NEURONS + j];
            }
            hidden_layer[j] = relu(sum);
        }

        double output_layer[OUTPUT_NEURONS];
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            double sum = output_bias[j];
            for (int k = 0; k < HIDDEN_NEURONS; k++) {
                sum += hidden_layer[k] * output_weights[k * OUTPUT_NEURONS + j];
            }
            output_layer[j] = sigmoid(sum);
        }

        // Round the output to 0 or 1
        int predicted = (output_layer[0] > 0.5) ? 1 : 0;
        int actual = (int)training_outputs[i][0];

        // Count correct predictions
        if (predicted == actual) {
            correct++;
        }

        // Print results for all 16 examples
        printf("Input: ");
        for (int j = 0; j < INPUT_NEURONS; j++) {
            printf("%d ", (int)training_inputs[i][j]);
        }
        printf("Output: %.4f, Predicted: %d, Actual: %d\n", 
               output_layer[0], predicted, actual);
    }

    // Print overall accuracy
    printf("Accuracy: %.2f%%\n", (float)correct / NUM_TRAINING_EXAMPLES * 100);

    return 0;
}
