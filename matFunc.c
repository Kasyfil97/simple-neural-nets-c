#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "matOps.h"
#include "matFunc.h"

LinearLayer createLinearLayer(int in, int out, bool bias){
    LinearLayer l;
    l.W = createMatrix(in, out, true);
    l.b = createMatrix(1, out, bias);
    return l;
}

void LinearForward(LinearLayer *layer, Matrix *X, Matrix *out){
    // shapeMatrix(X);
    // shapeMatrix(&layer->W);

    MatMul(X, &layer->W, out, true);
    for(int i=0;i<X->row;i++){
        for(int j=0;j<out->col;j++){
            out->data[i*out->col+j] += layer->b.data[j];
        }
    }
}

void freeLinearLayer(LinearLayer layer) {
    freeMatrix(&layer.W);
    freeMatrix(&layer.b);
}

void LinearBackward(LinearLayer *layer, Matrix *X, Matrix *dZ, Matrix *dX){
    Matrix Xt = createMatrix(X->col, X->row, 0);
    Transpose(X, &Xt);
    layer->dW = createMatrix(Xt.row, dZ->col, 0);
    MatMul(&Xt, dZ, &layer->dW, true);
    freeMatrix(&Xt);

    layer->db = createMatrix(1, dZ->col, 0);
    Sum(dZ, 0, &layer->db);

    Matrix Wt = createMatrix(layer->W.col, layer->W.row, 0);

    Transpose(&layer->W, &Wt);
    MatMul(dZ, &Wt, dX, true);
    freeMatrix(&Wt);
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

void Softmax(Matrix *m, Matrix *out, int axis){
    if(axis != 0 && axis != 1){
        printf("Softmax error: axis must be 0 or 1\n");
    }

    if(axis == 0){ // normalize per kolom
        for(int j=0; j<m->col; j++){
            float maxVal = -INFINITY;
            for(int k=0; k<m->row; k++){
                if(m->data[k*m->col+j] > maxVal)
                    maxVal = m->data[k*m->col+j];
            }
            float sumExp = 0.0f;
            for(int i=0; i<m->row; i++){
                sumExp += expf(m->data[i*m->col+j] - maxVal);
            }
            for(int i=0; i<m->row; i++){
                out->data[i*m->col+j] = expf(m->data[i*m->col+j] - maxVal) / sumExp;
            }
        }
    }
    else if(axis == 1){ // normalize per baris
        for(int i=0;i<m->row;i++){
            float max=-INFINITY;
            for(int j=0;j<m->col;j++){
                float v=m->data[i*m->col+j];
                if(v>max) max=v;
            }
            float sum=0;
            for(int j=0;j<m->col;j++){
                sum += expf(m->data[i*m->col+j]-max);
            }
            for(int j=0;j<m->col;j++){
                out->data[i*m->col+j] = expf(m->data[i*m->col+j]-max)/sum;
            }
        }
    }
    else{
        printf("Softmax: axis should be 0 for row or 1 for col");
    }
    out->col = m->col;
}


float CrossEntropyLoss(Matrix *pred, Matrix *label){
    float loss=0;
    for(int i=0;i<pred->row;i++){
        int c=(int)label->data[i];
        loss += -log(pred->data[i*pred->col+c]+1e-15);
    }
    return loss/pred->row;
}

void SoftmaxCrossEntropyBackward(Matrix *prob, Matrix *labels, Matrix *dZ){
    for(int i=0;i<prob->row;i++){
        int c=(int)labels->data[i];
        for(int j=0;j<prob->col;j++)
            dZ->data[i*prob->col+j]=prob->data[i*prob->col+j];
        dZ->data[i*prob->col+c]-=1.0f;
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