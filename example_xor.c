#define NN_IMPLEMENTATION
#include "nn_framework.h"

int
main(void)
{
    srand(time(0));

    //load XOR-data
    Mat ti, to;
    get_data("xor.txt", &ti, &to);

    // choose network architecture
    size_t arch[] = { ti.cols, 2, to.cols };
    double learning_rate = 1e-4;
    NN     nn     = CREATE_NN(arch, learning_rate);

    train(nn, ti, to, 5);       // 5k iterations
    evaluate(nn, ti, to);   // we evaluate on the training-set in this small example

    free(ti.es);   
    return 0;
}
