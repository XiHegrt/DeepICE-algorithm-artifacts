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
#include "deep_ice_chez.h"


void test_result(LABEL_TYPE* h_svs, LABEL_TYPE*h_t, int num_svs, int N, int8_t* asgns){
    int loss = 0;
    //for (int i = 0; i < num_svs; i++)
    //{
    //    for (int j = 0; j < N; j++)
    //    {
    //        std::cout << (int)h_svs[i * N + j] << ",";
    //    }
    //    std::cout << std::endl;
    //}

    for (int k = 0; k < pow(2, num_svs); k++) {
        for (int j = 0; j < N; j++) {
            int8_t temp = INT8_MIN;
            for (int i = 0; i < num_svs; i++)
            {
                temp = max(h_svs[i * N + j]*asgns[k*num_svs+i], temp);
            }
            loss += (LABEL_TYPE)((temp + h_t[j]) == 0);
        }
    std::cout<<"test_lose= "<<loss<<std::endl;
    loss = 0;
    }

}



extern __constant__ int8_t d_asgns[SIZE_ASGNS]; 
cnfg Chez_ICE(int N, int D, int K, DATA_TYPE* X, LABEL_TYPE* t) {
    /*****TEST***** */
    //LABEL_TYPE* test_svs = (LABEL_TYPE*)malloc(N*K*sizeof(LABEL_TYPE));

    // Step 1: Generate assignments
    int len = pow(2, K);
    int8_t *asgns = (int8_t *)malloc(len*K*sizeof(int8_t));
    asgns_gen_rec(asgns, len, K, K-1);

    checkCudaErrors(cudaMemcpyToSymbol(d_asgns, asgns, pow(2, NUM_RELUS)*K * sizeof(int8_t)));
    
    // Step 2: Prepare CUDA memory
    // TODO: optimize memory
    Workspace memPool = make_workspace(N, D, K);
    checkCudaErrors(cudaMemcpy(memPool.d_X, X, N * D * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(memPool.d_t, t, N * sizeof(LABEL_TYPE), cudaMemcpyHostToDevice));

    // Step 3: Iteratively generate hyperplanes and evaluate
    auto css = reserve_space_css(N, D);
    encoded_ncss ncss = make_ncss(K, 0,0);
    cnfg best_cnfg;
    best_cnfg.loss=INT16_MAX;


    //n is index of data points, which is the total number of data points used MINUS 1
    for (int n = 0; n < N; ++n) {
        std::cout << "This is stage " << n << std::endl;
        update_css(css, n);

        if (css.lens[css.max_r] > 0) {
            int n1 = combination(n, D);          // number of hyperplanes which have already been calculated
            int n2 = n1 + combination(n, D - 1); // equal to comb(n+1, D) // total number of hyperplanes existing in this stage

            if(memPool.limitN_svs_persistent > n){
                //store ws in temporary workspace

                size_t size_ws = (n2-n1)*D;
                int block_size = 256;
                int grid_size = size_ws>(block_size*64)? 64 : (size_ws+block_size-1)/block_size;
                init_array_datatype<<<grid_size, block_size>>>(memPool.d_hyper_ws, 1.0, size_ws);
                chez_gen_models_GPU(css.combs[D], css.lens[D], D, X, D, memPool.d_hyper_ws, memPool.d_Achunk);
                ws_to_svs(memPool.d_hyper_ws, n2-n1, memPool.d_X, N, D, memPool.d_dists, memPool.d_trans_dists, memPool.d_svs_persistent+n1*N);

            }
            css.lens[css.max_r] = 0; // empty

            update_ncss(ncss, n1, n2);

            if (ncss.num_choose_K > 0) {
                /****TEST CODE****/
                // show_ncss(ncss);
                // print_decoded_combs(ncss);

                auto temp_cnfg = blocked_evalfilt_GPU(ncss, K, N, D, memPool);

                if (temp_cnfg.loss < best_cnfg.loss) {
                        /*******TEST******/
                        //int cre =0;
                        //for(auto idx:temp_cnfg.comb){
                        //        cudaMemcpy(test_svs+N*cre, memPool.d_svs_persistent+idx*N, N*sizeof(LABEL_TYPE), cudaMemcpyDeviceToHost );
                        //        cre++;
                        //}
                        //test_result(test_svs, t, K, N, asgns);
                    best_cnfg = temp_cnfg;

                    if(best_cnfg.loss == 0){
                        return best_cnfg;
                    }

                    std::cout << "The best ncss is updated as: ";
                    for (auto &ele : best_cnfg.comb) {
                        std::cout << ele << ",";
                    }
                    std::cout << " it has 0-1 loss: " << best_cnfg.loss << std::endl;

                    //check svs 

                }
            }
        }
    }

    // Step 4: Cleanup
    free(asgns);
    free_workspace(memPool);
    free_combs(css);
    free_ncss(ncss);
    //extern unsigned *table_global;
    //free(table_global);
    //free(test_svs);

    return best_cnfg;
}



