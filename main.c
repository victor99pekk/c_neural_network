
#define NN_IMPLEMENTATION
#include "onlymatrix.h"

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

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = td
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td + 2
    };

    size_t arch[] = {2, 2, 1};
    NN nn = CREATE_NN(arch);
    NN g2 = CREATE_NN(arch);

    train(nn, g2, ti, to, 5);
    NN_PRINT(nn);
    nn_print_output(nn, ti, to);    //NN_PRINT(nn);

    printf("cost: %f\n", nn_cost(nn, ti, to));
    return 0;
}