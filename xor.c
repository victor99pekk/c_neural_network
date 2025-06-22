#define NN_IMPLEMENTATION
#include "nn_framework.h"

int
main(void)
{
    srand(time(0));

    /* ── load «xor.txt» ───────────────────────────────────────── */
    Mat ti, to;
    get_data("xor.txt", &ti, &to);          /* automatically sets rows / cols */

    /* network architecture matches the data file */
    size_t arch[] = { ti.cols, 2, to.cols };
    NN     nn     = CREATE_NN(arch, 1e-4);

    train(nn, ti, to, 5);                   /* 5 × 1000 iterations */
    nn_print_output(nn, ti, to);

    free(ti.es);                            /* one-and-done cleanup */
    return 0;
}
