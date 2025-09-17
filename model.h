#ifndef MODEL_H
#define MODEL_H

#include "matFunc.h"

typedef enum { LAYER_LINEAR, LAYER_RELU, LAYER_SOFTMAX } LayerType;

typedef struct {
    LayerType type;  // jenis layer (LINEAR, RELU, SOFTMAX, dll)
    void *layer;     // pointer ke data detail layer (misalnya LinearLayer)
} Layer;

typedef struct {
    Layer *layers; // array dari semua layer dalam model
    int num_layers;
    int capacity;
    Matrix *activations; // hasil keluaran tiap layer (untuk forward/backward)
} Model;

Model createModel(int capacity);
void addLayer(Model *model, LayerType type, int in_dim, int out_dim);

void modelForward(Model *model, Matrix *X, Matrix *buf1, Matrix *buf2, Matrix **out_probs);
void modelBackward(Model *model, Matrix *X_input, Matrix *Y_label, Matrix *buf1, Matrix *buf2);
void modelUpdate(Model *model, float lr);

void saveModel(Model *model, const char *filename);
Model loadModel(const char *filename, int num_layers);

void freeModel(Model *model);
void freeActivations(Model *model);

#endif