#define SIZE_SVS_LIMIT 1073741824 //2^30 bytes
// #define SIZE_DISTS_LIMIT 1048576
// #define SIZE_LOSSES_LIMIT 1048576
#define SIZE_COMBS_LIMIT  67108864 //2^26

#define FILT_WARP_SCALE 4 //16 warps per block
#define FILT_CNFG_STRIDE 64

struct Workspace{
    int N, D, K;
    DATA_TYPE* d_X;
    LABEL_TYPE* d_t;

    std::vector<std::vector<double>> shuffle_setting;

    int N_global;

    int limitN_svs_persistent;
    size_t capacity_svs_persistent; //for both old and new
    LABEL_TYPE*    d_svs_persistent;//after stored as svs, no need to keep corresponding ws and dists

    int *d_idxblock;
    int *d_mask;
    unsigned* d_cs_for_both;
    unsigned* d_cs_ordinary;
    unsigned* d_cs_reversed;

    DATA_TYPE* d_hyper_ws;
    DATA_TYPE* d_Achunk;

    size_t capacity_dists;
    DATA_TYPE* d_dists;
    DATA_TYPE* d_trans_dists;

    size_t capacity_combs_old;
    unsigned* h_combs_old;
    unsigned* d_combs_old;
    size_t capacity_combs_new;
    unsigned* h_combs_new;
    unsigned* d_combs_new;

    LOSS_TYPE* d_losses; 
    unsigned* d_losses_idx_old; 
    unsigned* d_losses_idx_new; 
    //int16_t* d_losses_pos_part; // length is max_numLosses
    //int16_t* d_losses_neg_part; // length is max_numLosses

    //reduce losses
    //int reduce_part_size;
    //int16_t *d_part_best_losses;
    //int* d_part_best_losses_ind;

};


struct encoded_ncss{
    unsigned K;
    // i is from [0, K), actually index 0 is always occupaied by 1, but for easy imagination, keep it
    unsigned *n_old; // number of combinaitons to pick i elements from C(N-1, D) 
    unsigned *n_new; // number of combinaitons pick K-i elements from C(N-1, D-1) 
                     //ncss_new[i] is represented by map(0~n_new[i]), and similar for ncss(before comcatenation)
    unsigned n1;
    unsigned n2;
    unsigned num_choose_K; // number of new combinations C(C(N, D), K) should be traverse

};
/* usage example
 *  to poick a combinaiton, given 3 ints, i, encoded_comb1, encoded_comb2
 *  return (map(encoded_comb1, n_old[i], i, comb1)) cancate (map(encoded_comb2, n_new[K-i], K-i, comb2))+C(N-1,D)
 */

Workspace make_workspace(int N, int D, int K);
Workspace make_workspace(int N, int D, int K, int global_N);
void free_workspace(Workspace& memPool);
encoded_ncss make_ncss(unsigned K, unsigned n1, unsigned n2);
void update_ncss(encoded_ncss& ncss, unsigned n1, unsigned n2);
void decode(unsigned *comb, encoded_ncss &ncss, int i, unsigned comb_old, unsigned comb_new);
void show_ncss(encoded_ncss& ncss);
void print_decoded_combs(encoded_ncss& ncss);
void free_ncss(encoded_ncss& ncss);
void batched_decode(unsigned* combs, unsigned x1, unsigned x2, unsigned n, unsigned r);
void test_batched_decode(unsigned N, unsigned D, unsigned K);
cnfg blocked_evalfilt_GPU(const encoded_ncss &ncss, int K, const int N, const int D, const Workspace &workspace);
void ws_to_svs(DATA_TYPE* d_ws, int num_ws, DATA_TYPE* d_X, int num_dataPoints, int D, DATA_TYPE* d_dists, DATA_TYPE *d_trans_dists, LABEL_TYPE* d_svs);
