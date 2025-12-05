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
