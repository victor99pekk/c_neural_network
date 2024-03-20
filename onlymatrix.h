#ifndef NN_H_
#define NN_H_

#include "../matrix_framework/matrix.h"
#include <stddef.h>
#include <assert.h>  // for assert
#include <stdio.h>
#include <time.h>
#include <math.h>

typedef struct{
    float learning_rate;
    int count;
    float m;
    float v;
    float beta1;
    float beta2;
} Adam;

typedef struct{
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // amount of activations is count +1
    Adam adam;
} NN;

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char *c);
void nn_rand(NN nn, float low, float high);
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g/* , float step */);
void nn_print_output(NN nn, Mat ti, Mat to);
void nn_symmetric(NN nn, float low, float high);

#define NN_RAND(nn) nn_rand(nn, 0, 3)
#define SYMMETRIC_INIT(nn) nn_symmetric(nn, 0, 3)
#define ARRAY_LEN(xs) (sizeof(xs) / sizeof((xs)[0]))
#define NN_PRINT(n) nn_print(nn, "neural network")
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#define CREATE_NN(arch) nn_alloc(arch, ARRAY_LEN(arch))
#define step learning_rate(nn.adam, gradient)


float learning_rate(Adam adam, float gradient)
{
    adam.m = adam.beta1 * adam.m + (1 - adam.beta1) * gradient;
    float mk = adam.m / (1 - powf(adam.beta1, adam.count));
    adam.v = adam.beta2 * adam.v + (1 - adam.beta2) * gradient * gradient;
    float vk = adam.v / (1 - powf(adam.beta2, adam.count));
    float div = sqrtf(adam.v) + 1e-8;
    return adam.learning_rate * (mk / div);
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("\n%s\n", name);
    Mat *ws = nn.ws;
    Mat *bs = nn.bs;
    for(size_t i = 0; i < nn.count; i++){
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(bs[i], buf, 4);
    }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
    matrix_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;

    nn.ws = (Mat*)matrix_MALLOC(sizeof(*nn.ws)*nn.count);
    matrix_ASSERT(nn.ws != NULL);
    nn.bs = (Mat*)matrix_MALLOC(sizeof(*nn.bs)*nn.count);
    matrix_ASSERT(nn.bs != NULL);
    nn.as = (Mat*)matrix_MALLOC(sizeof(*nn.as)*(nn.count+1));
    matrix_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for(size_t i = 0; i < nn.count; i++){
        nn.ws[i] = mat_alloc(nn.as[i].cols, arch[i+1]);
        nn.bs[i] = mat_alloc(1, arch[i+1]);
        nn.as[i+1] = mat_alloc(1, arch[i+1]);
    }

    nn.adam = (Adam){
        .learning_rate = 1e-4,
        .count = 1,
        .m = 0,
        .v = 0,
        .beta1 = 0.9,
        .beta2 = 0.999
    };

    SYMMETRIC_INIT(nn);

    return nn;
}

void nn_rand(NN nn, float low, float high)
{
    size_t layers = nn.count;
    for(size_t i = 0; i < layers; i++){
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

void nn_forward(NN nn){
    for(size_t i = 0; i < nn.count; i++){
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);    // multiplying weights
        mat_sum(nn.as[i+1], nn.bs[i]);             // adding bias
        mat_sig(nn.as[i+1]);                      // activation function
    }
}

float nn_cost(NN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    float c = 0;

    for(size_t i = 0; i < n; i++){
        Mat input = mat_row(ti, i);
        Mat target = mat_row(to, i);

        mat_copy(NN_INPUT(nn), input);
        nn_forward(nn);
        size_t q = to.cols;
        for(size_t j = 0; j < q; j++){
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(target, 0, j);
            c += d*d;
        }
    }
    return c/n;
}

void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to)
{
    float saved;
    float org = nn_cost(nn, ti, to);
    for(size_t layers = 0; layers < nn.count; layers++){
        for(size_t r = 0; r < nn.ws[layers].rows; r++){
            for(size_t c = 0; c < nn.ws[layers].cols; c++){
                saved = MAT_AT(nn.ws[layers], r, c);
                MAT_AT(nn.ws[layers], r, c) += eps;
                MAT_AT(g.ws[layers], r, c) = (nn_cost(nn, ti, to) - org) / eps;
                MAT_AT(nn.ws[layers], r, c) = saved;
            }
        }

        for(size_t r = 0; r < nn.bs[layers].rows; r++){
            for(size_t c = 0; c < nn.bs[layers].cols; c++){
                saved = MAT_AT(nn.bs[layers], r, c);
                MAT_AT(nn.bs[layers], r, c) += eps;
                MAT_AT(g.bs[layers], r, c) = (nn_cost(nn, ti, to) - org) / eps;
                MAT_AT(nn.bs[layers], r, c) = saved;
            }
        }
    }
}

void nn_learn(NN nn, NN g)
{
    float rate;
    for(size_t layers = 0; layers < nn.count; layers++){
        for(size_t r = 0; r < nn.ws[layers].rows; r++){
            for(size_t c = 0; c < nn.ws[layers].cols; c++){
                float gradient = MAT_AT(g.ws[layers], r, c);
                MAT_AT(nn.ws[layers], r, c) -= step;
            }
        }
        for(size_t r = 0; r < nn.bs[layers].rows; r++){
            for(size_t c = 0; c < nn.bs[layers].cols; c++){
                float gradient = MAT_AT(g.bs[layers], r, c);
                MAT_AT(nn.bs[layers], r, c) -= step;
            }
        }
    }
    nn.adam.count++;
}

void nn_print_output(NN nn, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    for(size_t i = 0; i < n; i++){
        Mat input = mat_row(ti, i);
        Mat target = mat_row(to, i);
        mat_copy(NN_INPUT(nn), input);
        nn_forward(nn);
        size_t q = to.cols;

        printf("\nDATA-SAMPLE: %zu\n", i+1);
        mat_print(input, "input", 15);
        mat_print(NN_OUTPUT(nn), "output", 15);
        mat_print(target, "target", 15);
    }
    printf("cost: %f\n", nn_cost(nn, ti, to));
}

void train(NN nn, NN g2, Mat ti, Mat to, int iterations)
{
    iterations = iterations * 1000;
    for(size_t i = 0; i < iterations; i++){
        nn_finite_diff(nn, g2, 1e-3, ti, to);
        nn_learn(nn, g2);
    }

}

void nn_symmetric(NN nn, float low, float high)
{
    size_t layers = nn.count;
    for(size_t i = 0; i < layers; i++){
        mat_symmetric_rand(nn.ws[i], low, high);
        mat_symmetric_rand(nn.bs[i], low, high);
    }
}


#endif //NN_IMPLEMENTATION