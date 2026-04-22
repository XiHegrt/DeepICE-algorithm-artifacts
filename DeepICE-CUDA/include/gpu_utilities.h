
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

#define CHECK_CUBLAS(call)                                               \
    {                                                                    \
        const cublasStatus_t status = call;                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                           \
            std::cerr << "CUBLAS error: " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                                     \
        }                                                                \
    }



struct cnfg{
    std::vector<unsigned> comb;
    LOSS_TYPE loss;
};
__global__ void init_array_datatype(DATA_TYPE* array, DATA_TYPE init_val, int len);
void chez_gen_models_GPU(unsigned* combs, int num_combs, int len_of_one_comb, DATA_TYPE* h_data, int D, DATA_TYPE* d_hyper_ws, DATA_TYPE*d_Achunk);
__global__ void gen_svs(DATA_TYPE* array, DATA_TYPE threshold, int8_t* ksvs, int len);
__global__ void data_orchestrate_datatype(unsigned* cs, int len_comb, int num_cs, DATA_TYPE* data_src, int len_dataPoint, DATA_TYPE* data_dst);
void coreset_gen_models_GPU(unsigned* d_combs, int num_combs, int len_of_one_comb, DATA_TYPE* d_X, int D, DATA_TYPE* d_hyper_ws, DATA_TYPE*d_Achunk);
