/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
/*
 * relu definition that can be used by kernels
*/
__device__ float relu(float x) { return x > 0 ? x : 0; }



__global__ void hidden_layer_kernel(
    const float* matrix,
    const float* vector,
    const float* b,
    float* result,
    int row_num,
    int col_num
){
    
    int j = blockIdx.x;                     // one block per col
    int t_id = threadIdx.x;                  
    extern __shared__ float partial[];

    float partial_sum = 0.0;

    // Each thread processes some rows
    for (int i = t_id; i < row_num; i += blockDim.x) {
        partial_sum += vector[i] * matrix[i * col_num + j];
    }
    partial[t_id] = partial_sum;
    __syncthreads();

    // parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t_id < stride) {
            partial[t_id] += partial[t_id + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (t_id == 0) {
        result[j] = relu(partial[0] + b[j])
    }

}



/*

call kernel:
int threadsPerBlock = 256;
int numBlocks = num_rows;




i think the below code is correct, but need to compile and test it:

__global__ void mv_mult_kernel(float* matrix, float* vector, float* bais, int num_rows, int num_cols, float* result){
    
    int t_idx = threadIdx.x; // handles chuncks of rows
    int col = blockIdx.x; //one col of the matrix is processed per block
    int stride = blockDim.x;

    exern __shared__ float partial[];

    if(col >= num_cols){
        return;
    }

    //compute partial sum
    partial[t_idx] = (t_idx < num_rows) ? matrix[t_idx * num_cols + col] * vector[t_idx] : 0.0;
    __syncthreads();

    int offset = stride;

    while(offset < num_rows){
        partial[t_idx + offset] = (t_idx + offest < num_rows) ? matrix[(t_idx + offset) * num_cols + col] * vector[t_idx + offset] : 0.0;
        __syncthreads();
        if (t_idx == 0){
            offset += stride;
        }
        float sum = partial[2*t_idx] + partial[2*t_idx +1];
        __syncthreads();
        partial[t_idx] = sum;
    }
    __syncthreads();

    //Partial sum reduction
    for (int increment = blockDim.x / 2; increment > 0; increment >>=1){ // shift notation for dividing by 2
        if (t_idx < increment){
            partial[t_idx] += partial[t_idx + increment];
        }
        __syncthreads();
    }

    if(t_idx == 0){
        result[row] = partial[0]
    }
}
    */

/* original-ish train_model
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    //start of addition
    float *d_W1, *d_W2, *d_W3, *d_b1, *d_b2, *d_b3;
    cudaMalloc(&d_W1, SIZE*H1*sizeof(float));
    cudaMalloc(&d_W2, H1*H2*sizeof(float));
    cudaMalloc(&d_W3, H2*CLASSES*sizeof(float));
    cudaMalloc(&d_b1, H1*sizeof(float));
    cudaMalloc(&d_b2, H2*sizeof(float));
    cudaMalloc(&d_b3, CLASSES*sizeof(float));
    cudaMemcpy(d_W1, model->W1, SIZE*H1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, model->W2, H1*H2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, model->W3, H2*CLASSES*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, model->b1, H1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, model->b2, H2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, model->b3, CLASSES*sizeof(float), cudaMemcpyHostToDevice);

    //end of addition

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ---------- Forward ----------
            float h1[H1], h1a[H1];
            for (int j=0;j<H1;j++){
                h1[j]=model->b1[j];
                for (int i=0;i<SIZE;i++){
                    h1[j]+=train_data[n][i]*model->W1[i*H1+j];
                }
                h1a[j]=relu(h1[j]);
            }
            float h2[H2], h2a[H2];
            for (int j=0;j<H2;j++){
                h2[j]=model->b2[j];
                for (int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j];
                h2a[j]=relu(h2[j]);
            }
            float out[CLASSES], outa[CLASSES];
            for (int k=0;k<CLASSES;k++){
                out[k]=model->b3[k];
                for (int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k];
            }
            softmax(out,outa,CLASSES);

            // ---------- Loss ----------
            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa[k]+1e-8f);

            // ---------- Backprop ----------
            float delta3[CLASSES];
            for (int k=0;k<CLASSES;k++)
                delta3[k] = train_label[n][k]-outa[k];

            float delta2[H2];
            for (int j=0;j<H2;j++){
                float err=0;
                for (int k=0;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
                delta2[j]=err*drelu(h2a[j]);
            }

            float delta1[H1];
            for (int j=0;j<H1;j++){
                float err=0;
                for (int k=0;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
                delta1[j]=err*drelu(h1a[j]);
            }

            // ---------- Update ----------
            for (int j=0;j<H2;j++)
                for (int k=0;k<CLASSES;k++)
                    model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
            for (int k=0;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

            for (int j=0;j<H1;j++)
                for (int k=0;k<H2;k++)
                    model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
            for (int k=0;k<H2;k++) model->b2[k]+=LR*delta2[k];

            for (int i=0;i<SIZE;i++)
                for (int j=0;j<H1;j++)
                    model->W1[i*H1+j]+=LR*delta1[j]*train_data[n][i];
            for (int j=0;j<H1;j++) model->b1[j]+=LR*delta1[j];
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_W3); cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_b3);
}
    */