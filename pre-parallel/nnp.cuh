/* nnp.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Header file for neural network model and training functions
*/

#ifndef NNP_H
#define NNP_H

// Model structure for neural network with two hidden layers
typedef struct tagMODEL{
    float W1[SIZE*H1];
    float b1[H1];
    float W2[H1*H2];
    float b2[H2];
    float W3[H2*CLASSES];
    float b3[CLASSES];
} MODEL;

// Activation function and derivative
__host__ __device__ float relu(float x);
__host__ float drelu(float y);

//function prototypes
__host__ void softmax(float *z, float *out, int len);
__host__ void init_weights(float *w, int size);
__host__ void train_model(MODEL* model);
__host__ void save_model(MODEL* model);
__host__ void load_model(MODEL* model);
__host__ void predict(float *x, MODEL* model);

#endif