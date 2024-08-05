A simple, proof of concept neural network in C, mainly for strengthening my mathematical intuition.

V1 was fitting to the nonlinear XOR function. Used just two bit inputs, producing 4 outputs, and used only sigmoid activations. 

# V2

This version uses ReLU activations in the hidden layer, and uses sigmoid activation for the output layer to make sure we normalize between 0 and 1. Also provides an accuracy score between predicted and desired outputs.

## How To Run

This one is simple. No packages or libraries to pip install. Just create a directory to store the file, and name your file something like `neuralnetwork.c`. Paste in my code from `nnV2.c`.

To compile, cd into the directory, and: 

```bash
gcc -o neuralnetwork neuralnetwork.c -lm
```

To run: 

```bash
./neuralnetwork
```

## Your results should look something like this:

![cscratchnnV2](Output.png)

the outputs that come from either 0 and 0 or 1 and 1 are closer to 0, whereas the inputs 1 and 0 or 0 and 1 produce results closer to 1. It's working. 

In my code, you should see the following:

```c
void generate_training_data(double inputs[NUM_TRAINING_EXAMPLES][INPUT_NEURONS], 
                            double outputs[NUM_TRAINING_EXAMPLES][OUTPUT_NEURONS]) {

    // DESIRED OUTPUT FUNCTION.
    int output_pattern[16] = {0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0};
```

`int output_pattern[16]` is where I perscribed my desired 4 bit truth table outputs. It's an arbitrary sequence that requires lots of nonlinearities to fit to. 
