#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "types.h"
#include "auxf.h"
#include "gpu_utilities.h"
#include "chez_util.h"


__constant__ int8_t d_asgns[SIZE_ASGNS];
cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k,
                                  const DATA_TYPE           *alpha,
                                  const DATA_TYPE           *A, int lda,
                                  const DATA_TYPE           *B, int ldb,
                                  const DATA_TYPE           *beta,
                                  DATA_TYPE                 *C, int ldc) {
#ifdef DOUBLE_PRECISION
  return cublasDgemm(handle, transa, transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
#else
  return cublasSgemm(handle, transa, transb,
                           m, n, k,
                           alpha,
                           A, lda,
                           B, ldb,
                           beta,
                           C, ldc);
#endif
}

// sinle matrix operation like transpose
cublasStatus_t cublasXgeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const DATA_TYPE           *alpha,
                          const DATA_TYPE           *A, int lda,
                          const DATA_TYPE           *beta,
                          const DATA_TYPE           *B, int ldb,
                          DATA_TYPE           *C, int ldc){
#ifdef DOUBLE_PRECISION
  return cublasDgeam(handle, transa, transb,
                           m, n, 
                           alpha, A, lda,
                           beta, B, ldb,
                           C, ldc);
#else
  return cublasSgeam(handle, transa, transb,
                           m, n, 
                           alpha, A, lda,
                           beta, B, ldb,
                           C, ldc);
#endif
}

// Function to find the largest N such that C(N, D) < X
int find_largest_n(int X, int D) {
    int low = D, high = X + D, result = D - 1; // N must be at least D

    while (low <= high) {
        int mid = low + (high - low) / 2;
        unsigned C = combination(mid, D);

        if (C < X) {
            result = mid; // Update result to current mid
            low = mid + 1; // Try to find a larger N
        } else {
            high = mid - 1; // Decrease N
        }
    }

    return result;
}

Workspace make_workspace(int N, int D, int K){
    Workspace memPool;
    memPool.N = N;
    memPool.D = D;
    memPool.K = K;

    // Step 1: Copy input data set to device
    checkCudaErrors(cudaMalloc(&memPool.d_X, N * D * sizeof(DATA_TYPE)));
    checkCudaErrors(cudaMalloc(&memPool.d_t, N * sizeof(LABEL_TYPE)));
    memPool.N_global = N;

    // Step 2: Achunk workspace for computing ws
    int N1 = combination(N-1, D);
    int N2 = N1 + combination(N-1, D-1);//== (N, D)
    int max_size_Achunk = (N2-N1)*(D*D);
    checkCudaErrors(cudaMalloc((void**)&memPool.d_Achunk, max_size_Achunk*sizeof(DATA_TYPE)));

    // Step 2.5
    memPool.d_idxblock = nullptr;
    memPool.d_mask = nullptr;
    memPool.d_cs_for_both = nullptr;
    

    // Step 3: persistent svs space
    int max_lines = SIZE_SVS_LIMIT/(N*sizeof(LABEL_TYPE)); // max numer of lines under limit of svs
    int max_N = find_largest_n(max_lines, D);// max number of data points in iteration that can be stored in persistent
    memPool.limitN_svs_persistent = max_N>N? N: max_N; //when n is larger or equal to max_N, it needs to be sotred in transient
    memPool.capacity_svs_persistent = combination(memPool.limitN_svs_persistent, D)* N;
    checkCudaErrors(cudaMalloc(&memPool.d_svs_persistent, memPool.capacity_svs_persistent* sizeof(LABEL_TYPE)));


    if(memPool.limitN_svs_persistent >= N){
        //only need temporary workspace for ws and dists

        // Step 4: ws tempory workspace of ws for computing ws, when then will be converted into svs
        size_t max_size_ws = (N2-N1)*D;
        checkCudaErrors(cudaMalloc((void**)&memPool.d_hyper_ws, max_size_ws* sizeof(DATA_TYPE)));
        int block_size = 256;
        int grid_size = max_size_ws>(block_size*64)? 64 : (max_size_ws+block_size-1)/block_size;
        //for calculate normal vector, init as 1
        init_array_datatype<<<grid_size, block_size>>>(memPool.d_hyper_ws, 1.0, max_size_ws);

        // Step 5: dists tempory workspace for computing ws, when then will be converted into svs
        size_t max_dists_size = (N2-N1)* N;  //it temperarily store dists which then be converted into svs_persistent or _transient
        memPool.capacity_dists = max_dists_size;
        checkCudaErrors(cudaMalloc(&memPool.d_dists, memPool.capacity_dists* sizeof(DATA_TYPE)));
        checkCudaErrors(cudaMalloc(&memPool.d_trans_dists, memPool.capacity_dists* sizeof(DATA_TYPE)));
    }
    //TODO: when no enough space for svs
    else{
        //for now only consider svs can be all saved
        fprintf(stderr, "Error: No engough GPU memory for svs!\n");
        exit(EXIT_FAILURE);

        // Step 4: ws persistent workspace for the rest of ws 
        size_t max_size_ws = (combination(N, D)-combination(memPool.limitN_svs_persistent, D));
        checkCudaErrors(cudaMalloc((void**)&memPool.d_hyper_ws, max_size_ws* sizeof(DATA_TYPE)));
        int block_size = 256;
        int grid_size = max_size_ws>(block_size*64)? 64 : (max_size_ws+block_size-1)/block_size;
        //for calculate normal vector, init as 1
        init_array_datatype<<<grid_size, block_size>>>(memPool.d_hyper_ws, 1.0, max_size_ws);
    }


    // space for combs
    int old_combs_size = combination(N1, K-1); 
    memPool.capacity_combs_old = old_combs_size>SIZE_COMBS_LIMIT? SIZE_COMBS_LIMIT: old_combs_size;
    memPool.h_combs_old = (unsigned*)malloc(memPool.capacity_combs_old *sizeof(unsigned));
    checkCudaErrors(cudaMalloc(&memPool.d_combs_old, memPool.capacity_combs_old *sizeof(unsigned)));

    int new_combs_size = combination(N2-N1, K); 
    memPool.capacity_combs_new = new_combs_size>SIZE_COMBS_LIMIT? SIZE_COMBS_LIMIT: new_combs_size;
    memPool.h_combs_new = (unsigned*)malloc(memPool.capacity_combs_new *sizeof(unsigned));
    checkCudaErrors(cudaMalloc(&memPool.d_combs_new, memPool.capacity_combs_new *sizeof(unsigned)));
    //std::cout<<"check old/new combs size: "<<memPool.capacity_combs_old<<","<<memPool.capacity_combs_new<<std::endl;


    // prepare space for losses
    int size_losses = pow(FILT_CNFG_STRIDE*FILT_WARP_SCALE, 2);
    checkCudaErrors(cudaMalloc(&memPool.d_losses, size_losses * sizeof(LOSS_TYPE)));
    checkCudaErrors(cudaMalloc(&memPool.d_losses_idx_old, size_losses* sizeof(unsigned)));
    checkCudaErrors(cudaMalloc(&memPool.d_losses_idx_new, size_losses* sizeof(unsigned)));

    //prepare space for reducing array
   // memPool.reduce_part_size = REDUCE_BLOCK_SIZE;
   // checkCudaErrors(cudaMalloc((void**)&memPool.d_part_best_losses, memPool.reduce_part_size* sizeof(int16_t)));
   // checkCudaErrors(cudaMalloc((void**)&memPool.d_part_best_losses_ind, memPool.reduce_part_size* sizeof(int)));

    return memPool;

}

Workspace make_workspace(int N, int D, int K, int global_N){
    Workspace memPool;
    memPool.N = N; // number of data set used to generate ws in this block
    memPool.D = D;
    memPool.K = K;

    // Step 1: Copy input data set to device
    checkCudaErrors(cudaMalloc(&memPool.d_X, global_N * D * sizeof(DATA_TYPE)));
    checkCudaErrors(cudaMalloc(&memPool.d_t, global_N * sizeof(LABEL_TYPE)));
    memPool.N_global = global_N;

    // Step 2: Achunk workspace for computing ws
    int N1 = combination(N-1, D);
    int N2 = N1 + combination(N-1, D-1);//== (N, D)
    int max_size_Achunk = (N2-N1)*(D*D);
    checkCudaErrors(cudaMalloc((void**)&memPool.d_Achunk, max_size_Achunk*sizeof(DATA_TYPE)));

    // Step 2.5 for reversed combinaitons
    checkCudaErrors(cudaMalloc((void**)&memPool.d_idxblock, N*sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&memPool.d_mask, (N2-N1)*N*sizeof(int)));
    int size_for_both = (N2-N1) * N;// cs and reversed_cs add up to be N
    checkCudaErrors(cudaMalloc((void**)&memPool.d_cs_for_both, size_for_both*sizeof(unsigned)));
    memPool.d_cs_reversed = memPool.d_cs_for_both;
    memPool.d_cs_ordinary = memPool.d_cs_for_both + (N2-N1)*(N-D);

    // Step 3: persistent svs space
    int max_lines = SIZE_SVS_LIMIT/(global_N*sizeof(LABEL_TYPE)); // max numer of lines under limit of svs
    int max_N = find_largest_n(max_lines, D);// max number of data points in iteration that can be stored in persistent
    memPool.limitN_svs_persistent = max_N>N? N: max_N; //when n is larger or equal to max_N, it needs to be sotred in transient
    memPool.capacity_svs_persistent = combination(memPool.limitN_svs_persistent, D)* global_N;
    checkCudaErrors(cudaMalloc(&memPool.d_svs_persistent, memPool.capacity_svs_persistent* sizeof(LABEL_TYPE)));


    if(memPool.limitN_svs_persistent >= N){
        //only need temporary workspace for ws and dists

        // Step 4: ws tempory workspace of ws for computing ws, when then will be converted into svs
        size_t max_size_ws = (N2-N1)*D;
        checkCudaErrors(cudaMalloc((void**)&memPool.d_hyper_ws, max_size_ws* sizeof(DATA_TYPE)));
        int block_size = 256;
        int grid_size = max_size_ws>(block_size*64)? 64 : (max_size_ws+block_size-1)/block_size;
        //for calculate normal vector, init as 1
        init_array_datatype<<<grid_size, block_size>>>(memPool.d_hyper_ws, 1.0, max_size_ws);

        // Step 5: dists tempory workspace for computing ws, when then will be converted into svs
        size_t max_dists_size = (N2-N1)* global_N;  //it temperarily store dists which then be converted into svs_persistent or _transient
        memPool.capacity_dists = max_dists_size;
        checkCudaErrors(cudaMalloc(&memPool.d_dists, memPool.capacity_dists* sizeof(DATA_TYPE)));
        checkCudaErrors(cudaMalloc(&memPool.d_trans_dists, memPool.capacity_dists* sizeof(DATA_TYPE)));
    }
    //TODO: when no enough space for svs
    else{
        //for now only consider svs can be all saved
        fprintf(stderr, "Error: No engough GPU memory for svs!\n");
        exit(EXIT_FAILURE);

        // Step 4: ws persistent workspace for the rest of ws 
        size_t max_size_ws = (combination(N, D)-combination(memPool.limitN_svs_persistent, D));
        checkCudaErrors(cudaMalloc((void**)&memPool.d_hyper_ws, max_size_ws* sizeof(DATA_TYPE)));
        int block_size = 256;
        int grid_size = max_size_ws>(block_size*64)? 64 : (max_size_ws+block_size-1)/block_size;
        //for calculate normal vector, init as 1
        init_array_datatype<<<grid_size, block_size>>>(memPool.d_hyper_ws, 1.0, max_size_ws);
    }


    // space for combs
    int old_combs_size = combination(N1, K-1); 
    memPool.capacity_combs_old = old_combs_size>SIZE_COMBS_LIMIT? SIZE_COMBS_LIMIT: old_combs_size;
    memPool.h_combs_old = (unsigned*)malloc(memPool.capacity_combs_old *sizeof(unsigned));
    checkCudaErrors(cudaMalloc(&memPool.d_combs_old, memPool.capacity_combs_old *sizeof(unsigned)));

    int new_combs_size = combination(N2-N1, K); 
    memPool.capacity_combs_new = new_combs_size>SIZE_COMBS_LIMIT? SIZE_COMBS_LIMIT: new_combs_size;
    memPool.h_combs_new = (unsigned*)malloc(memPool.capacity_combs_new *sizeof(unsigned));
    checkCudaErrors(cudaMalloc(&memPool.d_combs_new, memPool.capacity_combs_new *sizeof(unsigned)));
    //std::cout<<"check old/new combs size: "<<memPool.capacity_combs_old<<","<<memPool.capacity_combs_new<<std::endl;


    // prepare space for losses
    int size_losses = pow(FILT_CNFG_STRIDE*FILT_WARP_SCALE, 2);
    checkCudaErrors(cudaMalloc(&memPool.d_losses, size_losses * sizeof(LOSS_TYPE)));
    checkCudaErrors(cudaMalloc(&memPool.d_losses_idx_old, size_losses* sizeof(unsigned)));
    checkCudaErrors(cudaMalloc(&memPool.d_losses_idx_new, size_losses* sizeof(unsigned)));

    //prepare space for reducing array
   // memPool.reduce_part_size = REDUCE_BLOCK_SIZE;
   // checkCudaErrors(cudaMalloc((void**)&memPool.d_part_best_losses, memPool.reduce_part_size* sizeof(int16_t)));
   // checkCudaErrors(cudaMalloc((void**)&memPool.d_part_best_losses_ind, memPool.reduce_part_size* sizeof(int)));

    return memPool;

}
void free_workspace(Workspace& memPool){
    checkCudaErrors(cudaFree(memPool.d_X));
    checkCudaErrors(cudaFree(memPool.d_t));
    if(memPool.d_idxblock != nullptr) checkCudaErrors(cudaFree(memPool.d_idxblock));
    if(memPool.d_mask != nullptr) checkCudaErrors(cudaFree(memPool.d_mask));
    if(memPool.d_cs_for_both != nullptr) checkCudaErrors(cudaFree(memPool.d_cs_for_both));
    checkCudaErrors(cudaFree(memPool.d_svs_persistent));
    checkCudaErrors(cudaFree(memPool.d_hyper_ws));
    checkCudaErrors(cudaFree(memPool.d_Achunk));
    checkCudaErrors(cudaFree(memPool.d_dists));
    checkCudaErrors(cudaFree(memPool.d_trans_dists));
    //free(memPool.h_start_comb);
    //checkCudaErrors(cudaFree(memPool.d_start_comb));
    checkCudaErrors(cudaFree(memPool.d_losses));
    checkCudaErrors(cudaFree(memPool.d_losses_idx_old));
    checkCudaErrors(cudaFree(memPool.d_losses_idx_new));
}

// Function to map k to a combination
void map_number_to_combination(int k, int n, int r, unsigned *comb) {

    if(k<0 || k>=combination(n,r)){
        fprintf(stderr, "Incorrect index to decode\n");
        exit(EXIT_FAILURE);
    }
    int current_k = k+1; // 1-based index
    int index = 0;

    for (int i = 0; i < n; i++) {
        if (r == 0) break;

        // Count combinations if i is chosen as the first element
        unsigned count = combination(n - i - 1, r - 1);

        if (current_k > count) {
            // Skip this number
            current_k -= count;
        } else {
            // Choose this number and move to the next element
            comb[index++] = i;
            r--;
        }
    }
}


encoded_ncss make_ncss(unsigned K, unsigned n1, unsigned n2)
{
    // n1 == combination(N-1, D), n2== combination(N-1, D-1)
    encoded_ncss ncss;
    ncss.K = K;
    ncss.n_old = (unsigned *)malloc(K * sizeof(unsigned));       // [0, K-1]
    ncss.n_new = (unsigned *)malloc((K + 1) * sizeof(unsigned)); // [0, K]

    // N is the real number of data points, not strating form zero
    for (int i = 0; i < K; i++)
    {
        ncss.n_old[i] = combination(n1, i);
    }

    for (int i = 0; i <= K; i++)
    {
        ncss.n_new[i] = combination(n2 - n1, i);
    }

    ncss.n1 = n1;
    ncss.n2 = n2;
    ncss.num_choose_K = 0;

    return ncss;
}

void update_ncss(encoded_ncss& ncss, unsigned n1, unsigned n2){
  // N is the real number of data points, not strating form zero
  for(int i=0; i<ncss.K; i++){
    ncss.n_old[i] = combination(n1, i);
  }

  for(int i=0; i<=ncss.K; i++){
    ncss.n_new[i] = combination(n2-n1, i);
  }
    ncss.n1 = n1;
    ncss.n2 = n2;

    int temp = 0;
    for(int i=0; i<ncss.K; i++){
        temp += ncss.n_old[i]*ncss.n_new[ncss.K-i];
    }
    ncss.num_choose_K = temp;
}

void decode(unsigned *comb, encoded_ncss &ncss, int i, unsigned comb_old, unsigned comb_new) {
   //comb should be with len of K 
   map_number_to_combination(comb_old, ncss.n1, i, comb);
   map_number_to_combination(comb_new, ncss.n2-ncss.n1, ncss.K-i, comb+i);
   for(int k=i; k<ncss.K; k++){
    comb[k] += ncss.n1;
   }
}


void test_batched_decode(unsigned N, unsigned D, unsigned K){
    int n1 = combination(N-1, D);
    int n2 = n1+ combination(N-1, D-1);
    auto ncss = make_ncss(K, n1, n2);
    update_ncss(ncss, n1, n2);

    unsigned * ncss_K = (unsigned*)malloc(ncss.num_choose_K*K*sizeof(unsigned));
    unsigned *it_ncss_K=ncss_K;

    unsigned temp[K];
    clock_t begin = clock();
    for(int i=0; i<K; i++){
        for(int comb1=0; comb1<ncss.n_old[i]; comb1++){
            if(comb1>ncss.n_old[i]/2){
                std::cout<<"progress"<<std::endl;
            }
            map_number_to_combination(comb1, ncss.n1, i, temp);
            for(int comb2=0; comb2<ncss.n_new[K-i]; comb2++){
                map_number_to_combination(comb2, ncss.n2-ncss.n1, K-i, temp+i);
                for(int k=i; k<ncss.K; k++){
                    temp[k] += ncss.n1;
                }
                memcpy(it_ncss_K, temp, K*sizeof(unsigned));
                it_ncss_K += K;
            }
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Time spent for generate ((%d, %d), %d) combinaitons is %f s\n", N, D, K, time_spent);


    //for(int i=0; i<ncss.num_choose_K; i++){
    //    for(int j=0; j<K; j++){
    //        std::cout<<ncss_K[i*K+j]<<", ";
    //    }
    //    std::cout<<std::endl;
    //}
    free(ncss_K);
}

// Recursive function to generate combinations
//start: the smalles number that can be chosen
#define ACCESS_COL_MAJOR_MATRIX(arr, i, j, rows) arr[(j)*(rows)+(i)]
void generate_combinations_recursive(unsigned *combs, int level, int start, int n, int r, int current_index, int x1, int x2) {
    //end of recursion
    if (level == r-1) {
        //group size in this level is always 1
        unsigned long long group_size = n-start;

        int start_point = (current_index>=x1? current_index: x1) -x1;
        int end_point = (current_index + group_size >= x2? x2: current_index + group_size) -x1;
        int temp = start+(start_point+x1-current_index);
        //printf("n:%d, ", start_point);
        for(int j=start_point; j<= end_point; j++){
            //combs[j][level] = temp++;
            ACCESS_COL_MAJOR_MATRIX(combs, j, level, (x2-x1+1)) = temp++;
        }
        return;
    }

    // Iterate through possible values for the current level
    for (int i = start; i <= n - (r - level); i++) {
        //std::cout<<i<<", "<< level<<n<<r<<std::endl;
        unsigned long long group_size =combination(n - i - 1, r - level - 1);

        // Skip entire groups if out of range
        if (current_index+group_size-1 < x1) {
            current_index += group_size;
            continue;
        }
        
        //batched assign
        int start_point = (current_index >= x1? current_index: x1) - x1;
        int end_point = (current_index + group_size >= x2? x2: current_index + group_size) -x1;
        for(int j=start_point; j<= end_point; j++){
            ACCESS_COL_MAJOR_MATRIX(combs, j, level, (x2-x1+1)) = i;
        }

        // Recursively generate combinations in this group
        generate_combinations_recursive(combs, level + 1, i + 1, n, r, current_index, x1, x2);
        current_index += group_size;

        // Break early if we've reached the end of the range
        if(current_index> x2){
            break;
        }
    }
    return;
}

// Function to generate all combinations between x1 and x2
void batched_decode(unsigned* combs, unsigned x1, unsigned x2, unsigned n, unsigned r) {
    /*****TEST***** */
    //std::cout<<"batched: "<<x1<<","<<x2<<","<<n<<","<<r<<std::endl;
    if(r==0){
        return;
    }
    generate_combinations_recursive(combs, 0, 0, n, r, 0, x1, x2);
}



void show_ncss(encoded_ncss& ncss){
    printf("K: %d \nn1: %d; n2: %d", ncss.K, ncss.n1, ncss.n2);

    printf("\nn_old: ");
    for(int i=0; i<ncss.K; i++){
        printf("%d, ", ncss.n_old[i]);
    }

    printf("\nn_new: ");
    for(int i=0; i<ncss.K+1; i++){
        printf("%d, ", ncss.n_new[i]);
    }
    printf("\n\n");
}

void print_decoded_combs(encoded_ncss& ncss){
    unsigned comb[ncss.K];
    printf("K: %d \nn1: %d; n2: %d", ncss.K, ncss.n1, ncss.n2);

    printf("\nn_old: \n");
    for(int i=0; i<ncss.K; i++){
        for(int j=0; j<ncss.n_old[i]; j++){
            map_number_to_combination(j, ncss.n1, i, comb);
            for(int k=0; k<i; k++){
                printf("%d, ", comb[k]);
            }
            printf("\n");
        }
    }

    printf("\nn_new: \n");
    for(int i=0; i<=ncss.K; i++){
        for(int j=0; j< ncss.n_new[i]; j++){
            map_number_to_combination(j, ncss.n2-ncss.n1, i, comb);
            for(int k=0; k<i; k++){
                printf("%d, ", comb[k]+ncss.n1);
            }
            printf("\n");
        }
    }

    printf("\n\n");

}
void free_ncss(encoded_ncss& ncss){
    free(ncss.n_old);
    free(ncss.n_new);
}


__global__ void init_losses(LOSS_TYPE* array, LOSS_TYPE init_val, unsigned len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    for(int i = idx; i<len; i+=stride){
        array[i] = init_val;
    }
}


#define ACCESS_ROW_MAJOR_2D(array, cols, i, j) array[(i) * (cols) + (j)]
__global__ void calculate_decomposed_cnfg(LABEL_TYPE* d_svs, int num_data, LABEL_TYPE* d_t, unsigned* combs_old, unsigned* combs_new, int mark_i, unsigned num_old, unsigned num_new, LOSS_TYPE* best_losses, unsigned* best_losses_idx_old, unsigned*best_losses_idx_new){
        //printf("num_old: %d, mark_i: %d\n", num_old, mark_i);
    //one block for one configuration
    // one warp deal with one configuration
    const int num_threads_forOneconfi = blockDim.x;//one warp cooperate on N data points
    const int worker_idx = threadIdx.x; 
    const int idx_old =  blockIdx.x*blockDim.y+threadIdx.y; //for old
    const int idx_new =  blockIdx.y*blockDim.z+threadIdx.z; // for new
    const int stride_old = gridDim.x*blockDim.y;
    const int stride_new = gridDim.y*blockDim.z;

    LOSS_TYPE  * const loss_dst = best_losses+stride_old*idx_new+idx_old;
    unsigned *const loss_idx_old_dst = &ACCESS_ROW_MAJOR_2D(best_losses_idx_old, stride_old, idx_new, idx_old);
    unsigned *const loss_idx_new_dst = &ACCESS_ROW_MAJOR_2D(best_losses_idx_new, stride_old, idx_new, idx_old);

    unsigned losses_pos_loc_thread = 0;
    unsigned losses_neg_loc_thread = 0;

    // Level 1: traverse configurations
    for (int p_old = idx_old; p_old < num_old; p_old += stride_old) {
        for (int p_new = idx_new; p_new < num_new; p_new += stride_new) {
            LOSS_TYPE temp_during_asgns = INT16_MAX;

            // Level 2: traverse asgns
            for (int p_asgn = 0; p_asgn < pow(2, NUM_RELUS - 1); p_asgn++) {
                // Level 3: traverse data points
                for (int n = worker_idx; n < num_data; n += num_threads_forOneconfi) {
                    int8_t svs_loc = INT8_MIN;
                    // negative var
                    int8_t sum_ksvs_loc = 0;

                    // Level 4: traverse K svs
                    for(int reduce=0; reduce<mark_i; reduce++){
                        unsigned idx = ACCESS_COL_MAJOR_MATRIX(combs_old, p_old, reduce, num_old);
                        int8_t d_ksvs_loc = d_svs[idx * num_data + n];
                        svs_loc = max(svs_loc, d_ksvs_loc * d_asgns[p_asgn * NUM_RELUS + reduce]);

                        sum_ksvs_loc += d_ksvs_loc;
                    }
                    for(int reduce=mark_i; reduce<NUM_RELUS; reduce++){
                        unsigned idx = ACCESS_COL_MAJOR_MATRIX(combs_new, p_new, (reduce-mark_i), num_new);
                        int8_t d_ksvs_loc = d_svs[idx * num_data + n];
                        svs_loc = max(svs_loc, d_ksvs_loc * d_asgns[p_asgn * NUM_RELUS + reduce]);

                        sum_ksvs_loc += d_ksvs_loc;
                    }

                    int svs_neg_loc = svs_loc;
                    losses_pos_loc_thread += (unsigned)((svs_loc + d_t[n]) == 0);

                    if (sum_ksvs_loc == NUM_RELUS || sum_ksvs_loc == -NUM_RELUS) {
                        svs_neg_loc *= -1; 
                    }

                    losses_neg_loc_thread += (unsigned)((svs_neg_loc + d_t[n]) == 0);
                }
                // Perform iterative reduction within the warp
                for (int offset = 16; offset > 0; offset /= 2)
                {
                    losses_pos_loc_thread += __shfl_down_sync(0xFFFFFFFF, losses_pos_loc_thread, offset);
                    losses_neg_loc_thread += __shfl_down_sync(0xFFFFFFFF, losses_neg_loc_thread, offset);
                }

                // all threads in warp do, but only lane 0 has the global one
                temp_during_asgns = min(temp_during_asgns, min(losses_pos_loc_thread, losses_neg_loc_thread));

                losses_pos_loc_thread = 0;
                losses_neg_loc_thread = 0;
            }

            if (worker_idx == 0) {
                if (*loss_dst > temp_during_asgns) {
                    *loss_dst = temp_during_asgns;
                    *loss_idx_old_dst = p_old;
                    *loss_idx_new_dst = p_new;
                }
            }
        }
    }
}


cnfg GPU_reduce_2D(LOSS_TYPE* d_losses, unsigned* d_losses_idx_old, unsigned *d_losses_idx_new, const encoded_ncss &ncss, int mark_i, unsigned* h_combs_old, int num_combs_old, unsigned* h_combs_new, int num_combs_new){
    
    // Wrap the device pointer with thrust::device_ptr
    thrust::device_ptr<LOSS_TYPE> d_ptr = thrust::device_pointer_cast(d_losses);

    // Use thrust::min_element to find the smallest value
    thrust::device_ptr<LOSS_TYPE> min_ptr = thrust::min_element(d_ptr, d_ptr + pow(FILT_CNFG_STRIDE*FILT_WARP_SCALE, 2));

    // Get the smallest value
    LOSS_TYPE min_value = *min_ptr;
    //checkCudaErrors(cudaMemcpy(&min_value, min_ptr.get(), sizeof(LOSS_TYPE), cudaMemcpyDeviceToHost));

    // Calculate the 2D index of the smallest value
    int flat_index = min_ptr - d_ptr;


    //int size = pow(FILT_CNFG_STRIDE, 2);
    // LOSS_TYPE *h_array = new LOSS_TYPE[size];

    //// Copy data from GPU to host
    //checkCudaErrors(cudaMemcpy(h_array, d_losses, size * sizeof(LOSS_TYPE), cudaMemcpyDeviceToHost));
    //// Find the minimum on the host and its index
    //auto min_element_iter = std::min_element(h_array, h_array + size);
    //LOSS_TYPE min_value = *min_element_iter;
    //int flat_index = std::distance(h_array, min_element_iter);

    unsigned old_k, new_k;
    checkCudaErrors(cudaMemcpy(&old_k, d_losses_idx_old+flat_index, sizeof(unsigned), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&new_k, d_losses_idx_new+flat_index, sizeof(unsigned), cudaMemcpyDeviceToHost));
    

    std::vector<unsigned> comb(ncss.K);

    //map_number_to_combination(old_k, ncss.n1, mark_i, comb.data());
    //map_number_to_combination(new_k, ncss.n2-ncss.n1, ncss.K-mark_i, comb.data()+mark_i);
    for(int i=0; i<mark_i; i++){
        comb[i] = h_combs_old[num_combs_old*i + old_k];
    }
    for(int i=mark_i; i<NUM_RELUS; i++){
        comb[i] = h_combs_new[num_combs_new*(i-mark_i) + new_k];
    }
    for(int i=mark_i; i<ncss.K; i++){
        comb[i] += ncss.n1;
    }

    cnfg res;
    res.loss = min_value; 
    res.comb = comb;

    //delete[] h_array;

    return res;
}


__global__ void increment_combs_new(unsigned* combs_new, unsigned n1, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;

    for(int i = idx; i<len; i+=stride){
        combs_new[i] += n1;
    }
}




cnfg blocked_evalfilt_GPU(const encoded_ncss &ncss, int K, const int N, const int D, const Workspace &workspace) {
    cnfg res;
    res.loss = INT16_MAX;

    // Step 1: split N data points, can calculate big matrix, then pick to get losses

    // losses
    int size_losses = pow(FILT_CNFG_STRIDE*FILT_WARP_SCALE, 2);


    // Step 3: split ncss[K] by mark_i:
    for (int mark_i = 0; mark_i < K; mark_i++) { // each mark_i corresponds to a subset of all combs, namyly, calculte best losses in each subset, and finally get the best

        // TODO: generate combs onnhost and eval on GPU can overlap
        int num_old = ncss.n_old[mark_i];
        int num_new = ncss.n_new[K - mark_i];
        if(num_old<1 || num_new<1){
            continue; 
        }

        int stride_old;
        if(mark_i == 0){
            stride_old = num_old; // ==1
        }
        else{
            stride_old = workspace.capacity_combs_old/mark_i;
        }

        int stride_new = workspace.capacity_combs_new/(K-mark_i);

        for(int p_old=0; p_old<num_old; p_old+=stride_old){
            int end_old = p_old+stride_old>num_old? num_old: p_old+stride_old;
            int num_old_loc = end_old - p_old;
            batched_decode(workspace.h_combs_old, p_old, end_old-1, ncss.n1, mark_i);
            //{
            //    /******test *** */
            //    std::cout<<"OLD:"<<std::endl;
            //    printf("end_old:%d, num_old_loc:%d\n", end_old, num_old_loc);
            //    for (int i = 0; i < num_old_loc; i++)
            //    {
            //        for (int j = 0; j < mark_i; j++)
            //        {
            //            std::cout << workspace.h_combs_old[j * num_old_loc + i] << ", ";
            //        }
            //        std::cout << std::endl;
            //    }
            //}
            checkCudaErrors(cudaMemcpy(workspace.d_combs_old, workspace.h_combs_old, num_old_loc* mark_i * sizeof(unsigned), cudaMemcpyHostToDevice));
            
            for(int p_new=0; p_new<num_new; p_new+=stride_new){
                int end_new = p_new+stride_new>num_new? num_new: p_new+stride_new;
                int num_new_loc = end_new - p_new;
                batched_decode(workspace.h_combs_new, p_new, end_new-1, ncss.n2- ncss.n1, K-mark_i);

                //{
                //    /******test *** */
                //    std::cout<<"NEW:"<<std::endl;
                //    printf("end_new:%d, num_new_loc:%d\n", end_new, num_new_loc);
                //    for (int i = 0; i < num_new_loc; i++)
                //    {
                //        for (int j = 0; j < K-mark_i; j++)
                //        {
                //            std::cout << workspace.h_combs_new[j * num_new_loc + i] << ", ";
                //        }
                //        std::cout << std::endl;
                //    }
                //}

                checkCudaErrors(cudaMemcpy(workspace.d_combs_new, workspace.h_combs_new, num_new_loc* (K-mark_i) * sizeof(unsigned), cudaMemcpyHostToDevice));
                increment_combs_new<<<64, 1024>>>(workspace.d_combs_new, ncss.n1, num_new_loc*(K-mark_i));
                

                //filt configurations
                const int init_block_size = 1024;
                const int init_grid_size = size_losses > init_block_size * 96 ? 96 : (size_losses + init_block_size - 1) / init_block_size;
                init_losses<<<init_grid_size, init_block_size>>>(workspace.d_losses, N, size_losses);

                dim3 eval_block_size(32, FILT_WARP_SCALE, FILT_WARP_SCALE);  
                dim3 eval_grid_size(FILT_CNFG_STRIDE, FILT_CNFG_STRIDE);

                // one warp deal with one configuraiton
                calculate_decomposed_cnfg<<<eval_grid_size, eval_block_size>>>(workspace.d_svs_persistent, N, workspace.d_t,
                                                                            workspace.d_combs_old, workspace.d_combs_new, mark_i, num_old_loc, num_new_loc,
                                                                            workspace.d_losses, workspace.d_losses_idx_old, workspace.d_losses_idx_new);
                // find the best conf among this set of confguration
                auto temp_cnfg = GPU_reduce_2D(workspace.d_losses, workspace.d_losses_idx_old, workspace.d_losses_idx_new, ncss, mark_i, workspace.h_combs_old, num_old_loc, workspace.h_combs_new, num_new_loc);

                // update best configuration
                if (res.loss > temp_cnfg.loss) {
                    res = temp_cnfg;
                }


            }
        }
        

    }

    return res;
}

void ws_to_svs(DATA_TYPE* d_ws, int num_ws, DATA_TYPE* d_X, int num_dataPoints, int D, DATA_TYPE* d_dists, DATA_TYPE *d_trans_dists, LABEL_TYPE* d_svs){
    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("> ERROR: cuBLAS initialization failed\n");
    }

    DATA_TYPE alpha = 1.0f, beta = 1.0f;

    size_t size_dists = num_ws * num_dataPoints;
    int block_size = 256;
    int grid_size = size_dists > (block_size * 64) ? 64 : (size_dists + block_size - 1) / block_size;
    init_array_datatype<<<grid_size, block_size>>>(d_dists, -1, size_dists);
    CHECK_CUBLAS(cublasXgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             num_ws, num_dataPoints, D,
                             &alpha,
                             d_ws, D,
                             d_X, D,
                             &beta,
                             d_dists, num_ws));

    
    // Transpose svs by transpose dists, for better cache performance
    beta = 0.0f;
    CHECK_CUBLAS(cublasXgeam(
        handle,
        CUBLAS_OP_T,        // Transpose operation for input matrix
        CUBLAS_OP_N,        // No transpose for the second matrix (not used here)
        num_dataPoints, num_ws,         // Dimensions of the transposed matrix
        &alpha, d_dists, num_ws, // Input matrix
        &beta, d_trans_dists, num_dataPoints, // Output matrix
        d_trans_dists, num_dataPoints      // Result stored in d_output
    ));


// calculate svs
#ifdef DOUBLE_PRECISION
    DATA_TYPE threshold = 1e-12;
#else
    DATA_TYPE threshold = 1e-3;
#endif
    gen_svs<<<grid_size, block_size>>>(d_trans_dists, threshold, d_svs, size_dists);
    cublasDestroy(handle);
}


