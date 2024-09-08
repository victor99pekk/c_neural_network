
#define NN_IMPLEMENTATION
#include "nn_framework.h"

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};



int main(void)
{
    srand(time(0));
    size_t stride = 3;

    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Mat ti = get_training_data(td, stride, 2, n);
    Mat to = get_training_data(td, stride, 1, n);

    size_t arch[] = {2, 2, 1};
    NN nn = CREATE_NN(arch, 1e-4);

    train(nn, ti, to, 5);
    NN_PRINT(nn);
    nn_print_output(nn, ti, to);    //NN_PRINT(nn);

    print_cost(nn, ti, to);
    return 0;
}