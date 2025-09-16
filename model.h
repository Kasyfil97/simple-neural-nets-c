#ifndef MODEL_H
#define MODEL_H

#include "matFunc.h"

typedef enum { LAYER_LINEAR, LAYER_RELU, LAYER_SOFTMAX } LayerType;

typedef struct {
    LayerType type;
    void *layer; // pointer ke LinearLayer
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
    Matrix *activations;   // array pointer ke buffer aktivasi
} Model;

Model createModel();
void addLayer(Model *model, LayerType type, int in_dim, int out_dim);

void modelForward(Model *model, Matrix *X, Matrix *buf1, Matrix *buf2, Matrix **out_probs);
void modelBackward(Model *model, Matrix *X_input, Matrix *Y_label, Matrix *buf1, Matrix *buf2);
void modelUpdate(Model *model, float lr);

void saveModel(Model *model, const char *filename);
Model loadModel(const char *filename);

void freeModel(Model *model);
void freeActivations(Model *model);

#endif