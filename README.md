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
