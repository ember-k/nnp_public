/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */

__device__ float d_relu(float x) { return x > 0 ? x : 0; }
__device__ float d_drelu(float x) { return x > 0.0f ? 1.0f : 0.0f; }

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

    if (t_id == 0) {
        result[j] = d_relu(partial[0] + b[j]);
    }

}
/*
 * Matrix - Vector multiplication + bias
 * does't output the relu() value, but the pre-relu value.
 */
__global__ void output_layer_kernel(
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

    if (t_id == 0) {
        result[j] = partial[0] + b[j];
    }

}

__global__ void hidden_delta_kernel(
    const float* matrix,
    const float* delta_v,
    const float* prev_activation,
    float* delta_result,    
    int row_num, 
    int col_num              
){
    int j = blockIdx.x;   
    int t_id = threadIdx.x;
    extern __shared__ float partial[];

    float err = 0.0;

    for (int k = t_id; k < col_num; k += blockDim.x) {
        err += matrix[j * col_num + k] * delta_v[k];
    }

    partial[t_id] = err;
    __syncthreads();

    // parallel reduction 
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (t_id < stride)
            partial[t_id] += partial[t_id + stride];
        __syncthreads();
    }

    if (t_id == 0) {
        delta_result[j] = partial[0] * d_derelu(prev_activation[j]);
    }
}

__global__ void weight_update_kernel(
    float* matrix,        
    const float* prev_activation, 
    const float* delta_v,
    float lr,
    int row_num,
    int col_num
){
    int j = blockIdx.x;
    int t_id = threadIdx.x;

    for (int i = t_id; i < row_num; i += blockDim.x) {
        matrix[i * col_num + j] += lr * delta_v[j] * prev_activation[i];
    }
}


__global__ void bias_update_kernel(
    float* b, 
    const float* delta_v,
    float lr,
    int col_num
){
    int j = threadIdx.x;
    if (j < col_num) {
        b[j] += lr * delta_v[j];
    }
}


