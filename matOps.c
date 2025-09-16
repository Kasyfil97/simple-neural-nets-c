#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include "matOps.h"
#include <Accelerate/Accelerate.h>

Matrix createMatrix(int row, int col, bool fillRandom){
    Matrix matrix;
    matrix.row = row;
    matrix.col = col;
    matrix.data = (float*)malloc(row * col * sizeof(float));
    
    if(fillRandom){
        for(int i = 0; i < row*col; i++){
            matrix.data[i] = 2.0f * ((float)rand()/RAND_MAX) - 1.0f;
        }
    }
    else{
        for(int i = 0; i < row*col; i++){
            matrix.data[i] = 0.0f;
        }
    }
    return matrix;
}

void Transpose(Matrix *m, Matrix *out){
    out->row = m->col;
    out->col = m->row;
    for(int i = 0; i < m->row; i++){
        for(int j = 0; j < m->col; j++){
            out->data[IDX(out,j,i)] = m->data[IDX(m,i,j)];
        }
    }
}

void MatMul(Matrix *m1, Matrix *m2, Matrix *out, bool useAccelerate){
    if (m1->col != m2->row) {
        printf("MatMul error: dimension mismatch with first matrix col %d and second matrix row %d\n", m1->col, m2->col);
    }
    out->row = m1->row;
    out->col = m2->col;

    if(useAccelerate){
        if(m1->row <= 0 || m1->col <= 0 || m2->col <= 0){
            printf("MatMul warning: matrix dimensions invalid for BLAS with matrix 1 %d x %d and matrix 2 col %d, returning empty matrix\n", m1->row, m1->col, m2->col);
        }
        float alpha=1.0f, beta=0.0f;
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            m1->row, m2->col, m1->col,
            alpha,
            m1->data, m1->col,
            m2->data, m2->col,
            beta,
            out->data, out->col
        );
    }

    else{
        for (int i = 0; i < m1->row; i++){
                int rowA = i * m1->col;
                int rowC = i * out->col;
                for (int k = 0; k < m1->col; k++) {
                    float a = m1->data[rowA + k];
                    int rowB = k * m2->col;
                    for (int j = 0; j < m2->col; j++) {
                        out->data[rowC + j] += a * m2->data[rowB + j]; // Hapus baris M.data[rowC + j] = 0.0f;
                    }
                }
            }
        }
    
}


void Add(Matrix *m1, Matrix *m2, Matrix *out){
    if(m1->row != m2->row || m1->col != m2->col){
        printf("Add error: size mismatch\n");
    }

    for(int i = 0; i < m1->row * m1->col; i++){
        out->data[i] = m1->data[i] + m2->data[i];
        }
}

void Sum(Matrix *m, int axis, Matrix *out){
    if(axis==0){ // sum per kolom
        for(int j=0;j<m->col;j++){
            float s=0.0f;
            for(int i=0;i<m->row;i++){
                s += m->data[IDX(m,i,j)];
            }
            out->data[j] = s;
        }
    } else { // sum per baris
        for(int i=0;i<m->row;i++){
            float s=0.0f;
            for(int j=0;j<m->col;j++){
                s += m->data[IDX(m,i,j)];
            }
            out->data[i] = s;
        }
    }
}

void elemMul(Matrix *m1, Matrix *m2, Matrix *out){
    if(m1->row != m2->row || m1->col != m2->col){
        printf("ElemMul error: size mismatch\n");
    }

    for(int i = 0; i < m1->row * m1->col; i++){
            out->data[i] = m1->data[i] * m2->data[i]; 
    }
}

void scalarMul(Matrix *m, float scalar, Matrix *out){
    for(int i = 0; i < m->row * m->col; i++){
            out->data[i] = m->data[i] * scalar;
    }
}

void freeMatrix(Matrix *m){
    free(m->data);
    m->data = NULL;
}

void printMatrix(Matrix *m){
    for (int i = 0; i < m->row; i++) {
        for (int j = 0; j < m->col; j++) {
            printf("%f ", m->data[IDX(m, i, j)]);
        }
        printf("\n");
    }
}

void shapeMatrix(Matrix *m){
    printf("Shape: (%d, %d)\n", m->row, m->col);
}
