#include <stdio.h>
#include <stdlib.h>
#include "matOps.h"
#include "matFunc.h"
#include "model.h"

int main() {
    // === Load model ===
    Model model = loadModel("trained_model.bin");

    // === Tentukan dimensi input dari model ===
    int INPUT_DIM = model.layers[0].type == LAYER_LINEAR ? 
                    ((LinearLayer*)model.layers[0].layer)->W.row : 0;
    int batch = 5;  // jumlah sample untuk inference

    // === Load input ===
    // Di sini saya isi random dummy, bisa diganti baca file input.txt
    Matrix X_input = createMatrix(batch, INPUT_DIM, 1);

    // === Buat buffer reusable ===
    // Cari dimensi terbesar dari semua layer linear
    int max_dim = 0;
    for (int i = 0; i < model.num_layers; i++) {
        if (model.layers[i].type == LAYER_LINEAR) {
            LinearLayer *lin = (LinearLayer*)model.layers[i].layer;
            if (lin->W.col > max_dim) {
                max_dim = lin->W.col;
            }
        }
    }

    Matrix buf1 = createMatrix(batch, max_dim, 0);
    Matrix buf2 = createMatrix(batch, max_dim, 0);

    // === Forward pass ===
    Matrix *probs;
    modelForward(&model, &X_input, &buf1, &buf2, &probs);

    // === Prediksi kelas (argmax) ===
    int *preds = ArgMax(probs);

    for (int i = 0; i < batch; i++) {
        printf("sample %d: predicted class = %d\n", i, preds[i]);
    }

    // === Cleanup ===
    free(preds);
    freeMatrix(&X_input);
    freeMatrix(&buf1);
    freeMatrix(&buf2);
    freeModel(&model);

    return 0;
}
