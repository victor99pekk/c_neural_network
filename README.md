# c_neural_network
Neural network framework created to easily create various kinds of neural networks in c.

### Create a network:
Specify all hyperparameters for the neural network.
1. number of hidden layers, and number of nodes in the hidden layers
2. learning rate
```c
size_t arch[] = {2, 2, 1}; // (1)
learning_rate = 1e-4;      // (2)

NN nn = CREATE_NN(arch, learning_rate);
```
<br>

### Create training data:
define the training data. You have to define the a set of input to the neural network and it's expected output.

1. `stride` is how man is the number input parameters plus the number of outputs.

`ti` is input data that the nn can learn to fit.

`to` is the expected output that the nn learns to fit as well as possible.

```c
float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

size_t stride = 3;
size_t n = sizeof(td)/sizeof(td[0])/stride;

Mat ti = get_training_data(td, stride, 2, sizeof(td));
Mat to = get_training_data(td, stride, 1, sizeof(td));
```


### Train the Network
the train function takes 4 parameters.
1. A NN-struct who's weights will change to fit the input data.
2. training input and output, `ti` and `to`.
3. number of thousands of iterations that you want to train trough the training-data set.
```c
//    (1)   (2)  (3)
train(nn, ti, to, 5);
```

### Analyze the network
helpful function to analyse the network are to print the network, and see what the network outputs with specific input.

1. `NN_PRINT(NN nn)` prints all the layers and all the weights of the network. This can be helpful when you're dealing with smaller networks. but not so helpful when dealing with large ones.

    a network train to solve XORis printed as below:
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
2. `nn_print_output(nn, ti, to)` prints the an input to the nn, the exptected output, and the value that the nn outputs.
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

print_cost(nn)      // prints: cost: 0.000000
```
