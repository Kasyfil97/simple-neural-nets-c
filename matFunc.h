#ifndef MATFUNC_H
#define MATFUNC_H

#include "matOps.h"

typedef struct {
    Matrix W, b;
    Matrix dW, db;
} LinearLayer;

LinearLayer createLinearLayer(int in, int out, bool bias);

void LinearForward(LinearLayer *layer, Matrix *X, Matrix *out);
void LinearBackward(LinearLayer *layer, Matrix *X, Matrix *dZ, Matrix *dX);
void Sum(Matrix *m, int axis, Matrix *out);
void ReLU(Matrix *m, Matrix *out);
void ReLUBackward(Matrix *dY, Matrix *X, Matrix *dX);

void Softmax(Matrix *m, Matrix *out, bool axis_col);
float CrossEntropyLoss(Matrix *pred, Matrix *label);
void SoftmaxCrossEntropyBackward(Matrix *prob, Matrix *labels, Matrix *dZ);

int* ArgMax(Matrix *m);

#endif
