#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "model.h"
#include "matOps.h"

Model createModel(int capacity){
    Model m;
    m.layers=malloc(capacity * sizeof(Layer));
    if (!m.layers) {
        perror("malloc failed");
        exit(1);
    };
    m.num_layers=0;
    m.activations=NULL;
    m.capacity = capacity;
    return m;
}

void addLayer(Model *model, LayerType type, int in_dim, int out_dim){
    if (model->num_layers >= model->capacity) {
        fprintf(stderr, "Error: exceeded preallocated layer capacity\n");
        exit(1);
    }
    Layer *L=&model->layers[model->num_layers];

    L->type=type;
    if(type==LAYER_LINEAR){
        LinearLayer *lin=malloc(sizeof(LinearLayer));
        *lin=createLinearLayer(in_dim,out_dim, true);
        L->layer=lin;
    } else {
        L->layer=NULL;
    }
    model->num_layers++;
}

void modelForward(Model *model, Matrix *X, Matrix *buf1, Matrix *buf2, Matrix **out_probs){
    if(model->activations==NULL){
        model->activations=malloc(model->num_layers*sizeof(Matrix));
    }

    Matrix *cur=X;
    Matrix *out;
    Matrix *swap;

    for(int i=0;i<model->num_layers;i++){
        Layer L=model->layers[i];
        // buffer tujuan
        if(cur==buf1) out=buf2; else out=buf1;

        if(L.type==LAYER_LINEAR){
            LinearLayer *lin=(LinearLayer*)L.layer;
            LinearForward(lin,cur,out);
        }
        else if(L.type==LAYER_RELU){
            ReLU(cur,out);
        }
        else if(L.type==LAYER_SOFTMAX){
            Softmax(cur,out,true);
        }
        model->activations[i]=*out; // copy struct

        // ganti buffer
        swap=cur;
        cur=out;
        out=swap;
    }
    *out_probs=cur;
}

void modelBackward(Model *model, Matrix *X_input, Matrix *Y_label, Matrix *buf1, Matrix *buf2){
    int L=model->num_layers;

    Matrix *dZ=buf1;
    SoftmaxCrossEntropyBackward(&model->activations[L-1],Y_label,dZ);

    for(int i=L-1;i>=0;i--){
        Layer layer=model->layers[i];
        Matrix *A_prev=(i==0)?X_input:&model->activations[i-1];

        if(layer.type==LAYER_LINEAR){
            LinearLayer *lin=(LinearLayer*)layer.layer;
            LinearBackward(lin,A_prev,dZ,buf2);
            Matrix *tmp=dZ; dZ=buf2; buf2=tmp;
        }
        else if(layer.type==LAYER_RELU){
            ReLUBackward(dZ,A_prev,buf2);
            Matrix *tmp=dZ; dZ=buf2; buf2=tmp;
            
        }
    }
}

void modelUpdate(Model *model,float lr){
    for(int i=0;i<model->num_layers;i++){
        Layer L=model->layers[i];
        if(L.type==LAYER_LINEAR){
            LinearLayer *lin=(LinearLayer*)L.layer;
            for(int j=0;j<lin->W.row*lin->W.col;j++)
                lin->W.data[j]-=lr*lin->dW.data[j];
            for(int j=0;j<lin->b.row*lin->b.col;j++)
                lin->b.data[j]-=lr*lin->db.data[j];
            freeMatrix(&lin->dW);
            freeMatrix(&lin->db);
        }
    }
}

void saveModel(Model *model,const char *filename){
    FILE *fp=fopen(filename,"wb");
    if(!fp){perror("saveModel");return;}
    fwrite(&model->num_layers,sizeof(int),1,fp);

    for(int i=0;i<model->num_layers;i++){
        Layer L=model->layers[i];
        fwrite(&L.type,sizeof(LayerType),1,fp);
        if(L.type==LAYER_LINEAR){
            LinearLayer *lin=(LinearLayer*)L.layer;
            fwrite(&lin->W.row,sizeof(int),1,fp);
            fwrite(&lin->W.col,sizeof(int),1,fp);
            fwrite(lin->W.data,sizeof(float),lin->W.row*lin->W.col,fp);
            fwrite(&lin->b.row,sizeof(int),1,fp);
            fwrite(&lin->b.col,sizeof(int),1,fp);
            fwrite(lin->b.data,sizeof(float),lin->b.row*lin->b.col,fp);
        }
    }
    fclose(fp);
}

Model loadModel(const char *filename, int num_layers){
    FILE *fp=fopen(filename,"rb");
    if(!fp){perror("loadModel");exit(1);}
    Model model=createModel(num_layers);
    int num;
    fread(&num,sizeof(int),1,fp);

    for(int i=0;i<num;i++){
        LayerType t;
        fread(&t,sizeof(LayerType),1,fp);
        if(t==LAYER_LINEAR){
            int Wr,Wc,Br,Bc;
            fread(&Wr,sizeof(int),1,fp);
            fread(&Wc,sizeof(int),1,fp);
            Matrix W=createMatrix(Wr,Wc,0);
            fread(W.data,sizeof(float),Wr*Wc,fp);
            fread(&Br,sizeof(int),1,fp);
            fread(&Bc,sizeof(int),1,fp);
            Matrix b=createMatrix(Br,Bc,0);
            fread(b.data,sizeof(float),Br*Bc,fp);
            LinearLayer *lin=malloc(sizeof(LinearLayer));
            lin->W=W; lin->b=b;
            model.layers=realloc(model.layers,(model.num_layers+1)*sizeof(Layer));
            model.layers[model.num_layers].type=LAYER_LINEAR;
            model.layers[model.num_layers].layer=lin;
            model.num_layers++;
        } else {
            model.layers=realloc(model.layers,(model.num_layers+1)*sizeof(Layer));
            model.layers[model.num_layers].type=t;
            model.layers[model.num_layers].layer=NULL;
            model.num_layers++;
        }
    }
    fclose(fp);
    return model;
}

void freeModel(Model *model){
    for(int i=0;i<model->num_layers;i++){
        if(model->layers[i].type==LAYER_LINEAR){
            LinearLayer *lin=(LinearLayer*)model->layers[i].layer;
            freeMatrix(&lin->W);
            freeMatrix(&lin->b);
            free(lin);
        }
    }
    free(model->layers);
    model->layers=NULL;
}

void freeActivations(Model *model){
    if(model->activations){
        free(model->activations);
        model->activations=NULL;
    }
}
