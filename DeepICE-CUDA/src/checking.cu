#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <algorithm>  // for std::shuffle
#include <chrono>
#include <random>     // for std::mt19937
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "auxf.h"
#include "gpu_utilities.h"
#include "chez_util.h"

extern __constant__ int8_t d_asgns[SIZE_ASGNS]; 
// pick data and A chunk and then get ws, and then check loss

#define ACCESS_ROW_MAJOR_2D(array, cols, i, j) array[(i) * (cols) + (j)]
__global__ void calculate_loss(LABEL_TYPE* d_svs, int num_data, LABEL_TYPE* d_t,  LOSS_TYPE* losses){
        //printf("num_old: %d, mark_i: %d\n", num_old, mark_i);
    //one warp for one configuration

    int half_len_losses = pow(2, NUM_RELUS-1);

    //const int num_threads_forOneconfi = blockDim.x;//one warp cooperate on N data points
    //const int worker_idx = threadIdx.x; 
    //const int idx_old =  blockIdx.x*blockDim.y+threadIdx.y; //for old
    //const int idx_new =  blockIdx.y*blockDim.z+threadIdx.z; // for new
    //const int stride_old = gridDim.x*blockDim.y;
    //const int stride_new = gridDim.y*blockDim.z;


    unsigned losses_pos_loc_thread = 0;
    unsigned losses_neg_loc_thread = 0;

    for (int p_asgn = 0; p_asgn < pow(2, NUM_RELUS - 1); p_asgn++) { // Level 3: traverse data points
        for (int i = threadIdx.x; i < num_data; i += blockDim.x) {
            int8_t svs_loc = INT8_MIN;
            // negative var
            int8_t sum_ksvs_loc = 0;

            // Level 4: traverse K svs
            for(int j=0; j<NUM_RELUS; j++){
                int8_t d_ksvs_loc = d_svs[j * num_data + i];
                svs_loc = max(svs_loc, d_ksvs_loc * d_asgns[p_asgn * NUM_RELUS + j]);

                sum_ksvs_loc += d_ksvs_loc;
            }

            int svs_neg_loc = svs_loc;
            losses_pos_loc_thread += unsigned((svs_loc + d_t[i]) == 0);

            if (sum_ksvs_loc == NUM_RELUS || sum_ksvs_loc == -NUM_RELUS) {
                svs_neg_loc *= -1; 
            }

            losses_neg_loc_thread += unsigned((svs_neg_loc + d_t[i]) == 0);
        }
        // Perform iterative reduction within the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            losses_pos_loc_thread += __shfl_down_sync(0xFFFFFFFF, losses_pos_loc_thread, offset);
            losses_neg_loc_thread += __shfl_down_sync(0xFFFFFFFF, losses_neg_loc_thread, offset);
        }

        if(threadIdx.x == 0) {
            losses[p_asgn] = losses_pos_loc_thread;
            losses[p_asgn + half_len_losses] = losses_neg_loc_thread;
        }

        losses_pos_loc_thread = 0;
        losses_neg_loc_thread = 0;
    }

}

std::vector<int16_t> checking_result(DATA_TYPE *X, int N, int D, int K, Workspace& memPool, std::vector<unsigned> combs){
                
                init_array_datatype<<<1, 32>>>(memPool.d_hyper_ws, 1.0, K*D);

                chez_gen_models_GPU(combs.data(), K, D, X, D, memPool.d_hyper_ws, memPool.d_Achunk);
    //transfer_dataset(memPool·d_Achunk, X, D, cnfg.combs, K, D); 
    //_solve(memPool.d_Achunk, D, memPool.d_hyper_ws, 1, K);


                ws_to_svs(memPool.d_hyper_ws, K, memPool.d_X, N, D, memPool.d_dists, memPool.d_trans_dists, memPool.d_svs_persistent);

                calculate_loss<<<1, 32>>>(memPool.d_svs_persistent, N, memPool.d_t, memPool.d_losses);
                std::vector<int16_t> h_losses(pow(2,K));
                checkCudaErrors(cudaMemcpy(h_losses.data(), memPool.d_losses, pow(2, K)*sizeof(int16_t), cudaMemcpyDeviceToHost));
                return h_losses;
}
