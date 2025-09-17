#ifndef MATOPS_H
#define MATOPS_H
#include <stdbool.h>

typedef struct {
    int row;
    int col;
    float *data;
} Matrix;

#define IDX(m, i, j) ((i)*(m)->col + (j))

Matrix createMatrix(int row, int col, bool fillRandom);
void freeMatrix(Matrix *m);

void MatMul(Matrix *m1, Matrix *m2, Matrix *out, bool useAccelerate);
void MatMulWithTranspose(Matrix *m1, Matrix *m2, Matrix *out, 
                         bool transA, bool transB, bool useAccelerate);
void Add(Matrix *m1, Matrix *m2, Matrix *out);
void elemMul(Matrix *m1, Matrix *m2, Matrix *out);
void scalarMul(Matrix *m, float scalar, Matrix *out);
void Transpose(Matrix *m, Matrix *out);

void printMatrix(Matrix *m);
void shapeMatrix(Matrix *m);

#endif
