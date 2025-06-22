# c_neural_network
Neural network framework created to easily create various kinds of neural networks in c.

## contents
- [c\_neural\_network](#c_neural_network)
  - [contents](#contents)
  - [Create a network:](#create-a-network)
  - [Create training data:](#create-training-data)
  - [Train the Network](#train-the-network)
  - [Analyze the network](#analyze-the-network)


## Create a network:
Specify all hyperparameters for the neural network.
1. number of hidden layers, and number of nodes in the hidden layers
2. learning rate
```c
size_t arch[] = {2, 2, 1}; // (1)
learning_rate = 1e-4;      // (2)

NN nn = CREATE_NN(arch, learning_rate);
```
<br>

## Create training data:
To create the training and test data for the network we use a data loader that converts a .txt file into a matrix containing the data. Each row of the matrix belongs to one data point.


Here "data.txt" looks should follow a framework where the first row contains 2 numbers, first is input dim, the second is the target dim. The training data text file would look like the following:
```plaintext
2 1
0 0 0
0 1 1
1 0 1
1 1 0
```


Load the training/test data:
```c
Mat ti, to;
get_data("data.txt", &ti, &to);
```


## Train the Network
the train function takes 4 parameters.
1. A NN-struct who's weights will change to fit the input data.
2. training input and output, `ti` and `to`.
3. number of thousands of iterations that you want to train trough the training-data set.
```c
//    (1)   (2)  (3)
train(nn, ti, to, 5);
```

## Analyze the network
helpful function to analyze the network are to print the network, and see what the network outputs with specific input.

1. `NN_PRINT(NN nn)` prints all the layers and all the weights of the network. This can be helpful when you're dealing with smaller networks. but not so helpful when dealing with large ones.

    a network train to solve XOR is printed as below:
```c
neural network
    ws0: 
        9.874990 -7.004849 
        0.194340 0.230251 

    bs0: 
        -4.907146 3.203434 

    ws1: 
        9.933985 
        -7.969771 

    bs1: 
        -1.029707 
```
<br>
2. nn_print_output(nn, ti, to) prints the an input to the nn, the exptected output, and the value that the nn outputs.
<br>

<br>
the following is printed (in the terminal) when the print output is called for a trained XOR network:

```c

DATA-SAMPLE: 1
            // x1 = 0, x2 = 0
               input: 
                   0.000000 0.000000 

               output: 
                   0.000181 

               target: 
                   0.000000 


DATA-SAMPLE: 2
            // x1 = 0, x2 = 1
               input: 
                   0.000000 1.000000 

               output: 
                   0.999827 

               target: 
                   1.000000 

// AND SO ON...

```
<br>

3. `nn_cost` returns the summed cost-value over all samples in the training data. The cost is defined as the difference between the network output and the target. it is a good function to use to see it the network has improved or not after changes have been made to it
```c
nn_cost(nn, ti, to) // returns a float number that is the cost

print_cost(nn, ti, to)      
// => prints: 

// cost: 0.000002
```
