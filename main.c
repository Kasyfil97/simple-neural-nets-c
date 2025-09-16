#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matOps.h"
#include "matFunc.h"
#include "model.h"

int main(){
    srand(time(0));

    int in_row=10;
    int INPUT_DIM=512;
    int H1=1028, H2=128, OUT=3;
    int EPOCHS=100;
    float LR=0.00001f;

    Matrix X=createMatrix(in_row,INPUT_DIM, 1);
    Matrix Y=createMatrix(in_row,1,0);

    FILE *fp=fopen("label.txt","r");
    if(!fp){perror("label");return 1;}
    int a,i=0;
    while(fscanf(fp,"%d",&a)==1 && i<in_row){
        Y.data[i]=(float)a; i++;
    }
    fclose(fp);

    Model model=createModel();
    addLayer(&model,LAYER_LINEAR,INPUT_DIM,H1);
    addLayer(&model,LAYER_RELU,0,0);
    addLayer(&model,LAYER_LINEAR,H1,H2);
    addLayer(&model,LAYER_RELU,0,0);
    addLayer(&model,LAYER_LINEAR,H2,OUT);
    addLayer(&model,LAYER_SOFTMAX,0,0);

    // buffer reuse
    Matrix buf1f=createMatrix(in_row, H1>H2? (H1>OUT?H1:OUT):(H2>OUT?H2:OUT), 0);
    Matrix buf2f=createMatrix(in_row, buf1f.col, 0);

    Matrix buf1b=createMatrix(in_row, H1>H2? (H1>OUT?H1:OUT):(H2>OUT?H2:OUT), 0);
    Matrix buf2b=createMatrix(in_row, buf1b.col, 0);

    for(int e=0;e<EPOCHS;e++){
        Matrix *probs;
        modelForward(&model,&X,&buf1f,&buf2f,&probs);

        float loss=CrossEntropyLoss(probs,&Y);
        printf("Epoch %d, Loss %f\n",e,loss);

        modelBackward(&model,&X,&Y,&buf1b,&buf2b);
        modelUpdate(&model,LR);
    }

    saveModel(&model,"trained_model.bin");

    freeActivations(&model);
    freeModel(&model);
    freeMatrix(&X);
    freeMatrix(&Y);
    freeMatrix(&buf1f);
    freeMatrix(&buf2f);
    freeMatrix(&buf1b);
    freeMatrix(&buf2b);
}
