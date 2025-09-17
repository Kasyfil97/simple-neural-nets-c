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

    int total = m->row * m->col;
    for(int idx = 0; idx < total; idx++){
        int i = idx / m->col;   // baris asal
        int j = idx % m->col;   // kolom asal

        int new_idx = j * out->col + i; // posisi di matrix hasil (row-major)
        out->data[new_idx] = m->data[idx];
    }
}

void MatMul(Matrix *m1, Matrix *m2, Matrix *out, bool useAccelerate){
    if (m1->col != m2->row) {
        fprintf(stderr, "MatMul error: dimension mismatch with first matrix col %d and second matrix row %d\n", m1->col, m2->col);
        exit(1);
    }

    out->row = m1->row;
    out->col = m2->col;

    if(useAccelerate){
        if(m1->row <= 0 || m1->col <= 0 || m2->col <= 0){
            fprintf(stderr, "MatMul warning: matrix dimensions invalid for BLAS with matrix 1 %d x %d and matrix 2 col %d, returning empty matrix\n", m1->row, m1->col, m2->col);
            exit(1);
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
        int total = out->row * out->col;
        for (int idx = 0; idx < total; idx++) {
            int i = idx / out->col;
            int j = idx % out->col;

            float sum = 0.0f;
            for (int k = 0; k < m1->col; k++) {
                float a = m1->data[i * m1->col + k];   // elemen (i,k)
                float b = m2->data[k * m2->col + j];   // elemen (k,j)
                sum += a * b;
            }
            out->data[idx] = sum;
        }
    }
}

void MatMulWithTranspose(Matrix *m1, Matrix *m2, Matrix *out, 
                         bool transA, bool transB, bool useAccelerate) {
    
    int M = transA ? m1->col : m1->row;
    int K1 = transA ? m1->row : m1->col;
    int K2 = transB ? m2->col : m2->row;
    int N = transB ? m2->row : m2->col;

    if (K1 != K2) {
        printf("MatMulWithTranspose error: dimension mismatch\n");
        return;
    }

    out->row = M;
    out->col = N;

    if (useAccelerate) {
        float alpha = 1.0f, beta = 0.0f;
        cblas_sgemm(
            CblasRowMajor,
            transA ? CblasTrans : CblasNoTrans,
            transB ? CblasTrans : CblasNoTrans,
            M, N, K1,
            alpha,
            m1->data, m1->col,
            m2->data, m2->col,
            beta,
            out->data, out->col
        );
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K1; k++) {
                    float a = transA ? m1->data[k * m1->col + i] 
                                     : m1->data[i * m1->col + k];
                    float b = transB ? m2->data[j * m2->col + k] 
                                     : m2->data[k * m2->col + j];
                    sum += a * b;
                }
                out->data[i * N + j] = sum;
            }
        }
    }
}


void Add(Matrix *m1, Matrix *m2, Matrix *out){
    if(m1->row != m2->row || m1->col != m2->col){
        fprintf(stderr, "Add error: size mismatch\n");
        exit(1);
    }

    for(int i = 0; i < m1->row * m1->col; i++){
        out->data[i] = m1->data[i] + m2->data[i];
        }
}

void Sum(Matrix *m, int axis, Matrix *out){
    if(axis==0){ // sum per kolom
        for(int j=0; j<m->col; j++) out->data[j] = 0.0f;
        for(int i=0; i<m->row * m->col; i++){
            int col = i%m->col;
            out->data[col] += m->data[i];
        }

        // for(int j=0;j<m->col;j++){
        //     float s=0.0f;
        //     for(int i=0;i<m->row;i++){
        //         s += m->data[IDX(m,i,j)];
        //     }
        //     out->data[j] = s;
        // }

    } else { // sum per baris
        for(int i=0; i<m->row; i++) out->data[i] = 0.0f;
        for(int j=0; j<m->row * m->col; j++){
            int row = j / m->row;
            out->data[row] += m->data[j];
        }
        // for(int i=0;i<m->row;i++){
        //     float s=0.0f;
        //     for(int j=0;j<m->col;j++){
        //         s += m->data[IDX(m,i,j)];
        //     }
        //     out->data[i] = s;
        // }
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
    for(int i = 0; i < m->row * m->col; i++){
        int col = i%m->col;
        printf("%f ", m->data[i]);
        if(col==m->col - 1) printf("\n");
    }

    // for (int i = 0; i < m->row; i++) {
    //     for (int j = 0; j < m->col; j++) {
    //         printf("%f ", m->data[IDX(m, i, j)]);
    //     }
    //     printf("\n");
    // }
}

void shapeMatrix(Matrix *m){
    printf("Shape: (%d, %d)\n", m->row, m->col);
}
