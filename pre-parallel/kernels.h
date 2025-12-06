/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

__global__ void hidden_layer_kernel(
    const float* matrix,
    const float* vector,
    const float* b,
    float* result,
    int row_num,
    int col_num
);

__global__ void output_layer_kernel(
    const float* matrix,
    const float* vector,
    const float* b,
    float* result,
    int row_num,
    int col_num
);

__global__ void hidden_delta_kernel(
    const float* matrix,
    const float* delta_v,
    const float* prev_activation,
    float* delta_result,    
    int row_num, 
    int col_num              
);

__global__ void weight_update_kernel(
    float* matrix,        
    const float* prev_activation, 
    const float* delta_v,
    float lr,
    int row_num,
    int col_num
);

__global__ void bias_update_kernel(
    float* b, 
    const float* delta_v,
    float lr,
    int col_num
);