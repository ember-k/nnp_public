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

call kernel:
int threadsPerBlock = 256;
int numBlocks = num_rows;


i think the below code is correct, but need to compile and test it:
*/
__global__ void mv_mult_kernel(float* matrix, float* vector, int num_rows, int num_cols, float* result){
    
    int t_idx = threadIdx.x; //col we are working with
    int row = blockIdx.x; //one row of the matrix is processed per block
    int stride = blockDim.x;

    exern __shared__ float partial[];
    shared long offset;

    //compute partial sum
    partial[t_idx] = (t_idx < num_cols) ? matrix[row * num_cols + t_idx] * vector[t_idx] : 0.0;
    
    if(t_idx == 0){
        offset = stride;
    }
    __syncthreads();

    while(offset < num_cols){
        partial[t_idx + offset] = (t_idx + offest < num_cols) ? matrix[row * num_cols + t_idx + offset] * vector[t_idx + offset] : 0.0;
        __syncthreads();
        if(t_idx == 0){
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
