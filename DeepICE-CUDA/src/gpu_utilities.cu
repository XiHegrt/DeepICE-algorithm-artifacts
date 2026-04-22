#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "types.h"
#include "gpu_utilities.h"



// wrapper around cublas<t>getrfBatched()
cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int n,
                                   DATA_TYPE* const A[], int lda, int* P,
                                   int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
  return cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#endif
}

// wrapper around cublas<t>getrsBatched()
cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n,
                                   int nrhs, DATA_TYPE* const A[],
                                   int lda, const int* P, DATA_TYPE* B[],
                                   int ldb, int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return cublasDgetrsBatched(handle, trans, n, nrhs, A, lda, P, B, ldb,
                             info, batchSize);
#else
  return cublasSgetrsBatched(handle, trans, n, nrhs, A, lda, P, B, ldb,
                             info, batchSize);
#endif
}

// wrapper around cublas<t>getrsBatched()
cublasStatus_t cublasXgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const DATA_TYPE           *alpha,
                                  const DATA_TYPE           *A, int lda,
                                  long long int          strideA,
                                  const DATA_TYPE           *B, int ldb,
                                  long long int          strideB,
                                  const DATA_TYPE           *beta,
                                  DATA_TYPE                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount) {
#ifdef DOUBLE_PRECISION
  return cublasDgemmStridedBatched(handle, transa, transb,
                                           m, n, k,
                                           alpha,
                                           A, lda, strideA,
                                           B, ldb, strideB,
                                           beta,
                                           C, m, strideC,
                                           batchCount);
#else
  return cublasSgemmStridedBatched(handle, transa, transb,
                                           m, n, k,
                                           alpha,
                                           A, lda, strideA,
                                           B, ldb, strideB,
                                           beta,
                                           C, m, strideC,
                                           batchCount);
#endif
}


int _solve(DATA_TYPE* d_Aarray, int n, DATA_TYPE* d_Barray, int nrhs, int batchSize) {

  // cuBLAS variables
  cublasStatus_t status;
  cublasHandle_t handle;

  DATA_TYPE* h_Aptr_array[batchSize];
  DATA_TYPE* h_Bptr_array[batchSize];

  // device variables
  DATA_TYPE** d_Aptr_array;
  DATA_TYPE** d_Bptr_array;


  int* d_pivotArray;
  int* d_AinfoArray;
  int* d_BinfoArray;

  // initialize cuBLAS
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS initialization failed\n");
    return (EXIT_FAILURE);
  }

//#ifdef DOUBLE_PRECISION
//  printf("> Using DOUBLE precision...\n");
//#else
//  printf("> Using SINGLE precision...\n");
//#endif

  // allocate memory for device variables
  checkCudaErrors(
      cudaMalloc((void**)&d_pivotArray, n * batchSize * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_AinfoArray, batchSize * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_BinfoArray, batchSize * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void**)&d_Aptr_array, batchSize * sizeof(DATA_TYPE*)));
  checkCudaErrors(
      cudaMalloc((void**)&d_Bptr_array, batchSize * sizeof(DATA_TYPE*)));

  // create pointer array for matrices
  for (int i = 0; i < batchSize; i++) h_Aptr_array[i] = d_Aarray + (i * n * n);
  for (int i = 0; i < batchSize; i++) h_Bptr_array[i] = d_Barray + (i * n);

  // copy pointer array to device memory
  checkCudaErrors(cudaMemcpy(d_Aptr_array, h_Aptr_array,
                             batchSize * sizeof(DATA_TYPE*),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_Bptr_array, h_Bptr_array,
                             batchSize * sizeof(DATA_TYPE*),
                             cudaMemcpyHostToDevice));

  // perform LU decomposition
  //printf("> Performing LU decomposition...\n");

  status = cublasXgetrfBatched(handle, n, d_Aptr_array, n, d_pivotArray,
                               d_AinfoArray, batchSize);

  //printf("> Calculating X matrix...\n");

  int info;
  status = cublasXgetrsBatched(handle, CUBLAS_OP_T, n, nrhs, d_Aptr_array, n,
                               d_pivotArray, d_Bptr_array, n, &info,
                               batchSize);

  // free device variables
  checkCudaErrors(cudaFree(d_Aptr_array));
  checkCudaErrors(cudaFree(d_Bptr_array));
  checkCudaErrors(cudaFree(d_AinfoArray));
  checkCudaErrors(cudaFree(d_BinfoArray));
  checkCudaErrors(cudaFree(d_pivotArray));

  // destroy cuBLAS handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS uninitialization failed...\n");
    return (EXIT_FAILURE);
  }

  return (EXIT_SUCCESS);
}


// print column-major matrix
void printMatrix(DATA_TYPE* mat, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%6.3f ", mat[(j * height) + i]);
    }
    printf("\n");
  }
  printf("\n");
}
void printMatrix_row_major(DATA_TYPE* mat, int height, int width) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      printf("%6.3f ", mat[(i * width) + j]);
    }
    printf("\n");
  }
  printf("\n");
}

//TODO: naive implementation, need to give more efficient memcpy solution
void transfer_dataset(DATA_TYPE* d_Achunk, DATA_TYPE* h_A, int D, unsigned* combs, int len, int r) {
      //std::cout<<d_Achunk<<","<<std::endl;
  for(int i=0; i<len; i++){
    for(int j=0; j<r; j++){
      unsigned index = combs[i*r+j];
      //std::cout<<i<<","<<j<<":"<<index<<","<<std::endl;
      checkCudaErrors(cudaMemcpy(d_Achunk, h_A+index*D, D*sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
      d_Achunk += D;
    }
  }
}


void chez_gen_models_GPU(unsigned* combs, int num_combs, int len_of_one_comb, DATA_TYPE* h_data, int D, DATA_TYPE* d_hyper_ws, DATA_TYPE*d_Achunk) {
    // Allocate device memory for output
    // Overall: Ax=B=1
  //size_t AmatSize = D * D * sizeof(DATA_TYPE);
  DATA_TYPE* d_Bchunk = d_hyper_ws;
  int batchSize = num_combs;

  transfer_dataset(d_Achunk, h_data, D, combs, batchSize, len_of_one_comb);

  _solve(d_Achunk, D, d_Bchunk, 1, batchSize);

}


__global__ void data_orchestrate_datatype(unsigned* cs, int len_comb, int num_cs, DATA_TYPE* data_src, int len_dataPoint, DATA_TYPE* data_dst){
  //considering K and D are normally not huge, the kernel should be launchd serially, and each thread transfer a K*D matrix in data_dst
  int idx = blockDim.x*blockIdx.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  while(idx<num_cs){
    unsigned* local_cs_base = cs+idx*len_comb;
    DATA_TYPE* local_dst_base = data_dst+idx*len_comb*len_dataPoint;
    for(int i=0; i<len_comb; i++){
      DATA_TYPE* local_src_base = data_src+local_cs_base[i]*len_dataPoint;
      for(int j=0; j<len_dataPoint; j++){
        local_dst_base[i*len_dataPoint+j] =local_src_base[j];
      }
    }
    idx+=stride;
  }

}

void coreset_gen_models_GPU(unsigned* d_combs, int num_combs, int len_of_one_comb, DATA_TYPE* d_X, int D, DATA_TYPE* d_hyper_ws, DATA_TYPE*d_Achunk) {
  //overloded for compitable with reversed version

    // Allocate device memory for output
    // Overall: Ax=B=1
  //size_t AmatSize = D * D * sizeof(DATA_TYPE);
  DATA_TYPE* d_Bchunk = d_hyper_ws;
  int batchSize = num_combs;

  int block_size = 256;
  int grid_size = (num_combs+block_size-1)/block_size;
  grid_size = grid_size>64? 64: grid_size;
  data_orchestrate_datatype<<<grid_size, 256>>>(d_combs, len_of_one_comb, num_combs, d_X, D, d_Achunk);
  //transfer_dataset(d_Achunk, h_data, D, combs, batchSize, len_of_one_comb);

  _solve(d_Achunk, D, d_Bchunk, 1, batchSize);

}


__global__ void init_array_datatype(DATA_TYPE* array, DATA_TYPE init_val, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    for(int i = idx; i<len; i+=stride){
        array[i] = init_val;
    }
}



__global__ void gen_svs(DATA_TYPE* array, DATA_TYPE threshold, int8_t* ksvs, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    DATA_TYPE neg_threshold = -threshold;

    for(int i = idx; i<len; i+=stride){
      ksvs[i] =  (int8_t)(array[i]> threshold) - (int8_t)(array[i] <neg_threshold);
    }
}


