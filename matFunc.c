#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "matOps.h"
#include "matFunc.h"

LinearLayer createLinearLayer(int in, int out, bool bias){
    LinearLayer l;
    l.W = malloc(sizeof(Matrix));
    l.b = malloc(sizeof(Matrix));
    l.dW = malloc(sizeof(Matrix));
    l.db = malloc(sizeof(Matrix));


    *l.W = createMatrix(in, out, true);
    *l.b = createMatrix(1, out, bias);

    *l.dW = createMatrix(in, out, 0);
    *l.db = createMatrix(1, out, 0);

    return l;
}

void LinearForward(LinearLayer *layer, Matrix *X, Matrix *out){
    MatMul(X, layer->W, out, true);
    for(int i=0;i<X->row;i++){
        for(int j=0;j<out->col;j++){
            out->data[i*out->col+j] += layer->b->data[j];
        }
    }
}

// void freeLinearLayer(LinearLayer *layer) {
//     freeMatrix(&layer->W);
//     freeMatrix(&layer->b);
// }

void LinearBackward(LinearLayer *layer, Matrix *X, Matrix *dZ, Matrix *dX){

    // dW = X^T * dZ
    MatMulWithTranspose(X, dZ, layer->dW, true, false, true);

    // db = sum(dZ, axis=0)
    Sum(dZ, 0, layer->db);

    // dX = dZ * W^T
    MatMulWithTranspose(dZ, layer->W, dX, false, true, true);
}

void ReLU(Matrix *m, Matrix *out){
    for(int i=0;i<m->row*m->col;i++)
        out->data[i] = m->data[i]>0? m->data[i] : 0.1f*m->data[i];
    out->col = m->col;
}

void ReLUBackward(Matrix *dY, Matrix *X, Matrix *dX){
    for(int i=0;i<X->row*X->col;i++)
        dX->data[i] = (X->data[i]>0)? dY->data[i] : 0.1f*dY->data[i];
    dX->col = dY->col;
}

void Softmax(Matrix *m, Matrix *out, bool axis_col){
    int total = m->row * m->col;

    if(axis_col){ // normalize per kolom
        for(int i=0; i<m->row; i++){
                // cari max
            float maxVal = -INFINITY;
            for(int idx=0; idx<total; idx++){
                if(idx / m->col == i){  // ambil elemen di baris i
                    float v = m->data[idx];
                    if(v > maxVal) maxVal = v;
                }
            }
            // hitung sum exp
            float sumExp = 0.0f;
            for(int idx=0; idx<total; idx++){
                if(idx / m->col == i){
                    sumExp += expf(m->data[idx] - maxVal);
                }
            }
            // assign hasil
            for(int idx=0; idx<total; idx++){
                if(idx / m->col == i){
                    out->data[idx] = expf(m->data[idx] - maxVal) / sumExp;
                }
            }
        }
    }
    else{ // normalize per baris
        for(int j=0; j<m->col; j++){
            // cari max
            float maxVal = -INFINITY;
            for(int idx=0; idx<total; idx++){
                if(idx % m->col == j){  // ambil elemen di kolom j
                    float v = m->data[idx];
                    if(v > maxVal) maxVal = v;
                }
            }
            // hitung sum exp
            float sumExp = 0.0f;
            for(int idx=0; idx<total; idx++){
                if(idx % m->col == j){
                    sumExp += expf(m->data[idx] - maxVal);
                }
            }
            // assign hasil
            for(int idx=0; idx<total; idx++){
                if(idx % m->col == j){
                    out->data[idx] = expf(m->data[idx] - maxVal) / sumExp;
                }
            }
        }
    }

    out->col = m->col;
}

float CrossEntropyLoss(Matrix *pred, Matrix *label){
    if(pred->row != label->row){
        fprintf(stderr, "crossentropy loss: the prediction row (%d) and label row (%d) is not same size", pred->row, label->row);
        exit(1);
    }
    float loss=0;
    for(int i=0;i<pred->row;i++){
        int c=(int)label->data[i];
        loss += -log(pred->data[i*pred->col+c]+1e-7);
    }
    return loss/pred->row;
}

void SoftmaxCrossEntropyBackward(Matrix *prob, Matrix *labels, Matrix *dZ){
    int total = prob->row * prob->col;

    for(int idx=0; idx<total; idx++){
        dZ->data[idx] = prob->data[idx];
    }

    for(int i=0; i<prob->row; i++){
        int c = (int)labels->data[i];
        int idx = i * prob->col + c;  // bisa juga ROW/COL macro
        dZ->data[idx] -= 1.0f;
    }

    dZ->col = prob->col;
}

int* ArgMax(Matrix *m){
    int *p=malloc(m->row*sizeof(int));
    for(int i=0;i<m->row;i++){
        int idx=0;
        float max=m->data[i*m->col];
        for(int j=1;j<m->col;j++){
            float v=m->data[i*m->col+j];
            if(v>max){max=v; idx=j;}
        }
        p[i]=idx;
    }
    return p;
}