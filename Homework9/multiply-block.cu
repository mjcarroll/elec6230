#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <sys/time.h>

#define BLOCK_SIZE 16 
#define MATRIX_SIZE 4096 
#define WINDOW_SIZE 2

bool InitCUDA(void)
{
        int count = 0;int i = 0;
        cudaGetDeviceCount(&count);
        if(count == 0) {
                fprintf(stderr, "There is no device.\n");
                return false;
        }
        for(i = 0; i < count; i++) {
                cudaDeviceProp prop;
                if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
                        if(prop.major >= 1) {
                                break;
        }}}
        if(i == count) {
                fprintf(stderr, "There is no device supporting CUDA.\n");
                return false;
        }
        cudaSetDevice(i);
        printf("CUDA initialized.\n");
        return true;
}

__global__ void MatMulKernel(float* Md, float* Nd, float* Pd)
{
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;

    float Pvalue = 0;
    
    for(int m = MATRIX_SIZE * BLOCK_SIZE * by, n = BLOCK_SIZE * bx;
            m <= MATRIX_SIZE * BLOCK_SIZE * by + MATRIX_SIZE -1;
            m += BLOCK_SIZE, n += BLOCK_SIZE * MATRIX_SIZE)
    {
        __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
        
        Mds[ty][tx] = Md[m + MATRIX_SIZE * ty + tx];
        Nds[ty][tx] = Nd[n + MATRIX_SIZE * ty + tx];
       
        // Make sure that all the threads have copied to shared memory before 
       // performing the actual multiplication
        __syncthreads();
        
#pragma unroll
        for( int k = 0; k < BLOCK_SIZE; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // Syncronize after the multiplication.
        __syncthreads();    
    }
    Pd[MATRIX_SIZE * BLOCK_SIZE * by + 
        BLOCK_SIZE * bx + 
        MATRIX_SIZE * ty 
        + tx] = Pvalue;
}

int main(int argc, char* argv[])
{
    struct timeval t0,t1;
    // Initialize CUDA using the ASC helper function
    if(!InitCUDA()) {
            return 0;
    }
    // Define some sizes for malloc
    unsigned int size = MATRIX_SIZE * MATRIX_SIZE;
    unsigned int mem_size = sizeof(float) * size;
    // Declare the variables to be used 
    float* A = (float*) malloc(mem_size);
    float* B = (float*) malloc(mem_size);
    float* C = (float*) malloc(mem_size);
    float* Md;
    float* Nd;
    float* Pd;
    // Initialize the A and B matricies to the homework specifications 
    int row,col;
    for( int i=0; i<size; i++)
    {
        row = i/MATRIX_SIZE;
        col = i%MATRIX_SIZE;
        A[i] = ((row + 1.0)*(col + 1.0))/MATRIX_SIZE;
        B[i] = (col + 1.0)/(row + 1.0);
    }
    
    gettimeofday(&t0,0);
    // Allocate the matricies on the video card
    cudaMalloc((void**) &Md, mem_size);
    cudaMalloc((void**) &Nd, mem_size);
    cudaMalloc((void**) &Pd, mem_size);
    // Copy the matricies to the video card
    cudaMemcpy(Md, A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, B, mem_size, cudaMemcpyHostToDevice);
    // Perform the Kernel 
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(MATRIX_SIZE/dimBlock.x,MATRIX_SIZE/dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(Md,Nd,Pd);
    // Copy the results
    cudaMemcpy(C, Pd, mem_size, cudaMemcpyDeviceToHost);
    // Clear the memory on the video card
    cudaFree(Md);cudaFree(Nd);cudaFree(Pd);

    gettimeofday(&t1,0);
    // Print a 16x16 "test section" to prove results are correct. 
    for( int i=0; i<16; i++){
        for( int j=0; j<16; j++){
            printf("%6.2f ",C[j*MATRIX_SIZE+i]);
        }
        printf("\n");
    }

    printf("\nTime Results\n");
    float totalInt = t1.tv_sec - t0.tv_sec + (t1.tv_usec - t0.tv_usec)*1.0E-06;
    printf("Total Execution Time:\t%e\n",totalInt);

    free(A);free(B);free(C);
    return 0;
}
