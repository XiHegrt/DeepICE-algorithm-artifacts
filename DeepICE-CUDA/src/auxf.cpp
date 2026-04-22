#include <iostream>
#include <cstring>
#include <vector>
#include <random>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include "types.h"
#include "auxf.h"
using namespace std;

// ----------------- Assignment Generation -----------------
void asgns_gen_rec(int8_t* asgns, int len, int K, int offset) {
    
    if(offset==-1){
        return; //end of recursion
    }

    for(int i=0; i<len/2; i++){
        asgns[offset+i*K] = 1;
    }
    for(int i=len/2; i<len; i++){
        asgns[offset+i*K] = -1;
    }
    asgns_gen_rec(asgns, len/2, K, offset-1);
    asgns_gen_rec(asgns+(len/2)*K, len/2, K, offset-1);
}

//unsigned *table_global;
//unsigned R_global;
//void genCombTable(unsigned N, unsigned R){
//    R_global = R;
//    table_global=(unsigned*)malloc((N+1)*(R+1)*sizeof(unsigned));
//    unsigned value = 1;
//    for(int n=0; n<N+1; n++){
//        for(int r=0; r<R+1; r++){
//            if(n==0){
//                table_global[n*(R+1)+r]=0;
//                continue;
//            }
//            if(r==0){
//                table_global[n*(R+1)+r]=1;
//                continue;
//            }
//            value *= (n-r+1);
//            value /= r;
//            table_global[n*(R+1)+r]=value;
//        }
//        value = 1;
//    }
//}

unsigned combination(unsigned n, unsigned r) {
    //return table_global[n*(R_global+1)+r];
    if (r > n) return 0;
    if (r == 0 || r == n) return 1;

    unsigned result = 1;
    for (unsigned i = 1; i <= r; ++i) {
        result *= (n-i+1);
        result /= i;
    }
    return result; 
}

css_whole reserve_space_css(int N, int D){
    // Init css
    css_whole css;
    css.max_n=N;
    css.max_r=D;
    css.lens = (unsigned*)malloc((css.max_r+1)*sizeof(unsigned));
    css.max_lens = (unsigned*)malloc((css.max_r+1)*sizeof(unsigned));

    int size_css=0;
    css.lens[0] = 1; //C(n, 0) == 1
    css.max_lens[0]=1;
    for(int i=1; i<css.max_r; i++){
        //reserve C(N, i) space
        int temp = combination(css.max_n, i);
        size_css += i*temp;
        css.max_lens[i]=temp; //store max lens temporarily
    }
    css.max_lens[css.max_r] = combination(css.max_n-1, css.max_r-1); 
    size_css += css.max_lens[css.max_r]*css.max_r; 
    css.whole_mem = (unsigned *)malloc(size_css*sizeof(unsigned));
    css.combs = (unsigned **)malloc((css.max_r+1)*sizeof(unsigned*));

    unsigned* base = css.whole_mem;
    css.combs[0] = base;
    for(int i=1; i<(css.max_r+1); i++){
        //reserve C(N, i) space
        css.combs[i] = base;
        base += i*css.max_lens[i];
        css.lens[i] = 0;
    }

    return css;

}

css_whole reserve_space_ncss(css_whole& css, int K) {
    // Init ncss
    css_whole ncss;
    ncss.max_n=combination(css.max_n, css.max_r);
    ncss.max_r=K;
    ncss.lens = (unsigned*)malloc((ncss.max_r+1)*sizeof(unsigned));
    ncss.max_lens = (unsigned*)malloc((ncss.max_r+1)*sizeof(unsigned));

    int n1 = combination(css.max_n-1, css.max_r);
    int n2 = n1 + combination(css.max_n-1, css.max_r-1);
    int size_ncss=0;
    ncss.lens[0] = 1;//C(n, 0) == 1
    ncss.max_lens[0] = 1;//C(n, 0) == 1
    for(int i=1; i<ncss.max_r; i++){
        int temp = combination(ncss.max_n, i);
        size_ncss += i*temp;
        ncss.max_lens[i]=temp; //store max lens temporarily
    }

    ncss.max_lens[ncss.max_r] = 0;
    for(int i=1; i<ncss.max_r; i++){
        ncss.max_lens[ncss.max_r] += combination(n2-n1, i)*combination(n1, ncss.max_r-i);
    }
    ncss.max_lens[ncss.max_r] += combination(n2-n1, ncss.max_r);
    size_ncss += ncss.max_lens[ncss.max_r]*ncss.max_r; 
    //std::cout<<"last len: "<<ncss.max_lens[ncss.max_r]*ncss.max_r<<std::endl;

    ncss.whole_mem = (unsigned *)malloc(size_ncss*sizeof(unsigned));
    ncss.combs = (unsigned **)malloc((ncss.max_r+1)*sizeof(unsigned*));

    unsigned* base = ncss.whole_mem;
    ncss.combs[0] = base;
    for(int i=1; i<(ncss.max_r+1); i++){
        //reserve C(N, i) space
        ncss.combs[i] = base;
        base += i*ncss.max_lens[i];
        ncss.lens[i] = 0;
    }
    return ncss;
}

void free_combs(css_whole& css) {
    free(css.combs);
    free(css.lens);
    free(css.whole_mem);
}
// ----------------- Vector Utilities -----------------
double dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

void normalize(std::vector<float>& vec) {
    float norm = std::sqrt(dotProduct(vec, vec)); // Compute the L2 norm
    for (float& val : vec) {
        val /= norm;
    }
}


// Function to generate a dataset
void generateDataset(int D, int N, double overlap, double balance_fac,
                     vector<vector<float>>& X, vector<int8_t>& t, unsigned int seed) {
    mt19937 gen(seed); // Seeded random number generator
    uniform_real_distribution<> dis(-0.5, 0.5);
    normal_distribution<> noise_dist(0.0, 1.0);

    bool accept = false;

    while (!accept) {
        // Initialize X (D x N) and labels t
        X.assign(N, vector<float>(D));
        t.resize(N);

        // Generate features
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
                X[i][j] = 2.0 * dis(gen); // Random values in range [-1, 1]
            }
        }

        // Generate random hyperplane
        float p0t = 0.8 * dis(gen);
        vector<float> pt(D);
        for (int i = 0; i < D; ++i) {
            pt[i] = 0.8 * dis(gen);
        }
        normalize(pt); // Normalize the hyperplane
        p0t /= sqrt(D); // Normalize p0t by sqrt(D)

        // Compute labels based on hyperplane
        int Npos = 0, Nneg = 0;
        for (int i = 0; i < N; ++i) {
            double decision_value = p0t;
            decision_value += dotProduct(pt, X[i]);
            decision_value += overlap * noise_dist(gen); // Add noise

            if (decision_value >= 0) {
                t[i] = 1; // Positive class
                ++Npos;
            } else {
                t[i] = -1; // Negative class
                ++Nneg;
            }
        }

        // Check if the class balance condition is satisfied
        accept = (Npos > balance_fac * N) && (Nneg > balance_fac * N);
    }
}

std::vector<float> linearize(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> linearData;
    for (const auto& row : matrix) {
        linearData.insert(linearData.end(), row.begin(), row.end());
    }
    return linearData;
}
// ----------------- Auxiliary Utilities -----------------
// Function to print a 2D vector (used for debugging)
void print_2D_vector(const std::vector<std::vector<unsigned>>& vec) {
    for (const auto& row : vec) {
        for (int elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}


void print_2D_array(const unsigned* arr, int rows, int cols) {
    for(int i=0;i<rows;i++){
        for(int j=0; j<cols; j++){
            std::cout<<arr[i*cols+j]<<", ";
        }
        std::cout<<std::endl;
    }
}
void update_css(css_whole &css, int n){
    //data point has index from 0, so when n-th stage, we actually have n+1 data points
    for(int d =1; d<std::min(css.max_r, n+1)+1; ++d){ // d is choosing d elements
        for(int i=0; i<css.lens[d-1]; i++){
            auto this_base_for_one_comb = css.combs[d]+(css.lens[d]+i)*(d);
            auto last_base_for_one_comb = css.combs[d-1]+(css.lens[d-1]-i-1)*(d-1); //flip
            for(int j=0; j<d-1; j++) {
                this_base_for_one_comb[j] = last_base_for_one_comb[j];
            }
            this_base_for_one_comb[d-1] = n; // right upd
        }
    }

    for(int d =std::min(css.max_r, n+1); d>0; --d){ // actually use d+1 as index
        css.lens[d] += css.lens[d-1];
    }
    

}
std::vector<std::vector<unsigned>> right_upd_array(int a, int M, const std::vector<std::vector<unsigned>>& cs) {
    if(cs[0].empty()){
        std::vector<std::vector<unsigned>> cs_new={{}};
        cs_new[0].push_back(static_cast<unsigned>(a));
        return cs_new;
    }
    else{
        std::vector<std::vector<unsigned>> cs_new(cs.size());
        std::reverse_copy(cs.begin(), cs.end(), cs_new.begin());
        for(int i=0; i<M; i++){
            cs_new[i].push_back(static_cast<unsigned>(a));
        }
        return cs_new;
    }
}

std::vector<std::vector<unsigned>> upd_array(int a, int M, const std::vector<std::vector<unsigned>>& cs) {
    if(cs[0].empty()){
        std::vector<std::vector<unsigned>> cs_new={{}};
        cs_new[0].push_back(static_cast<unsigned>(a));
        return cs_new;
    }
    else{
        std::vector<std::vector<unsigned>> cs_new(cs);
        for(int i=0; i<M; i++){
            cs_new[i].push_back(static_cast<unsigned>(a));
        }
        return cs_new;
    }
}

std::vector<::vector<std::vector<unsigned>>> kcombs_iter_array(int K, int n1, int n2) {
    std::vector<std::vector<unsigned>> emt;
    emt.push_back(std::vector<unsigned>{});
    std::vector<std::vector<std::vector<unsigned>>> cngs(K + 1, emt); // Unused in this implementation

    for (int n = 0; n < n2-n1; ++n) {
        std::vector<std::vector<std::vector<unsigned>>> temp(cngs);
        for (int k = 0; k < std::min(K, n+1); ++k) {
            int M = combination(n, k);

            if (temp[k + 1][0].empty()) {
                // first time reach this d(rightest in this level), so temp[d+1] is empty
                cngs[k+1] = upd_array(n+n1, M, temp[k]);//to avoid redundant compuatation, no need 
                //print_2D_vector(css[d + 1]);
                //print_2D_vector(temp[d]);
            } else {
                auto updated = upd_array(n+n1, M, temp[k]);
                cngs[k + 1].insert(cngs[k + 1].end(), updated.begin(), updated.end());
            }
        }
    }
    //std::cout<<"sss"<<cngs.size()<<std::endl;
    //print_2D_vector(cngs[1]);
    return cngs;
}


css_whole new_kcombs_iter_array(int n1, int n2, int K) {
    // Prepare the ncss_new
    css_whole ncss;
    ncss.max_n = n2-n1;
    ncss.max_r = K;
    ncss.lens = (unsigned*)malloc((ncss.max_r+1)*sizeof(unsigned));
    ncss.max_lens = (unsigned*)malloc((ncss.max_r+1)*sizeof(unsigned));

    int size_ncss=0;
    ncss.lens[0] = 1;
    ncss.max_lens[0] = 1;
    for(int i=1; i<ncss.max_r+1; i++){ // ncss_new has last combination FULL
        //reserve C(N, i) space
        int temp = combination(ncss.max_n, i);
        size_ncss += i*temp;
        ncss.max_lens[i]=temp; //store max lens temporarily
    }
    ncss.whole_mem = (unsigned *)malloc(size_ncss*sizeof(unsigned));
    ncss.combs = (unsigned **)malloc((ncss.max_r+1)*sizeof(unsigned*));

    unsigned* base = ncss.whole_mem;
    ncss.combs[0] = base;
    for(int i=1; i<(ncss.max_r+1); i++){
        //reserve C(N, i) space
        ncss.combs[i] = base;
        base += i*ncss.max_lens[i];
        ncss.lens[i] = 0;
    }


    // Calculate the ncss_new
    //whole combinations from [n1, n2)
    for (int n = 0; n < n2-n1; ++n) {
        for(int k =1; k<std::min(ncss.max_r, n+1)+1; ++k){ // d is choosing d elements
            for(int i=0; i<ncss.lens[k-1]; i++){
                auto this_base_for_one_comb = ncss.combs[k]+(ncss.lens[k]+i)*k;
                auto last_base_for_one_comb = ncss.combs[k-1]+i*(k-1); //no need to flip
                for(int j=0; j<k-1; j++) {
                    this_base_for_one_comb[j] = last_base_for_one_comb[j];
                }
                this_base_for_one_comb[k-1] = n+n1; // right upd
            }
        }

        for(int k =std::min(ncss.max_r, n+1); k>0; --k){ // actually use d+1 as index
            ncss.lens[k] += ncss.lens[k-1];
        }
    }

    return ncss;
}


cs cross_join_array(cs& cs1, cs& cs2) {
    if(cs1[0].empty() || cs2[0].empty()){
        return cs{{}};
    }
    int num_cols = cs1[0].size()+cs2[0].size();
    int num_rows = cs1.size()*cs2.size();

    cs result(num_rows);
    for(auto& row : result){
        row.reserve(num_cols);
    }

    if(cs1.size()<cs2.size()){
        for(int i=0;i<cs1.size();i++){
            for(int j=0;j<cs2.size();j++){
                vector<unsigned> &row = result[i*cs2.size()+j];
                row.insert(row.end(), cs1[i].begin(), cs1[i].end());
                row.insert(row.end(), cs2[j].begin(), cs2[j].end());
            }
        }
    }
    else{
        for(int i=0;i<cs2.size();i++){
            for(int j=0;j<cs1.size();j++){
                vector<unsigned> &row = result[i*cs1.size()+j];
                row.insert(row.end(), cs1[j].begin(), cs1[j].end());
                row.insert(row.end(), cs2[i].begin(), cs2[i].end());
            }
        }
    }

    //std::cout<<"s1"<<std::endl;
    //print_2D_vector(cs1);
    //std::cout<<"s2"<<std::endl;
    //print_2D_vector(cs2);
    //print_2D_vector(result);
    return result;

}


std::vector<cs> convol_filt_array(cs (*f)(cs&, cs&), int k, vector<cs> x, vector<cs> y){
    vector<cs> result(k+1, {{}});
    int len_x = x.size();
    int len_y = y.size();

    for(int i=0; i<len_x; i++){
        for(int j=0; j<len_y; j++){
            if((i+j)>k){
                continue;
            }
            if(i==0){
                if(y[j][0].empty()){
                    continue;
                }
                else if(result[i+j][0].empty()){
                    result[i+j] = y[j];
                }
                else{
                    result[i+j].insert(result[i+j].end(), y[j].begin(), y[j].end());
                }
            }
            else if(j==0){
                if(x[i][0].empty()){
                    continue;
                }
                else if(result[i+j][0].empty()){
                    result[i+j] = x[i];
                }
                else{
                    result[i+j].insert(result[i+j].end(), x[i].begin(), x[i].end());
                }
            }
            else{
                auto cs_new = f(x[i], y[j]);
                if(cs_new[0].empty()){
                    continue;
                }
                else if(result[i+j][0].empty()){
                    result[i+j] = cs_new;
                }
                else{
                    result[i+j].insert(result[i+j].end(), cs_new.begin(), cs_new.end());
                }
            }

        }
    }

    return result;
}






CSVResult readCSVtoRowMajorAndLastColumn(const std::string& fileName, size_t& rows, size_t& cols) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + fileName);
    }

    std::vector<DATA_TYPE> rowMajorArray;
    std::vector<LABEL_TYPE> lastColumn;
    std::string line;
    rows = 0;
    cols = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        size_t currentCols = 0;
        double lastValue = 0;
        std::vector<DATA_TYPE> rowValues;

        while (std::getline(ss, cell, ',')) {
            DATA_TYPE value = std::stod(cell);
            rowValues.push_back(value);
            currentCols++;
        }

        if (rows == 0) {
            cols = currentCols; // First row determines the number of columns
        } else if (currentCols != cols) {
            throw std::runtime_error("Inconsistent number of columns in row " + std::to_string(rows + 1));
        }

        // Separate the last column and row-major values
        lastValue = rowValues.back();
        rowValues.pop_back(); // Remove the last value from the row values
        rowMajorArray.insert(rowMajorArray.end(), rowValues.begin(), rowValues.end());
        lastColumn.push_back((LABEL_TYPE)lastValue);

        rows++;
    }

    cols--; // Adjust column count to exclude the last column
    file.close();
    return {rowMajorArray, lastColumn};
}

void new_cross_join(unsigned *result, unsigned* x, int len_x, int stride_x, unsigned* y, int len_y, int stride_y){
    for(int i=0; i<len_x; i++){
        for(int j=0; j<len_y; j++){
            std::memcpy(result, x+i*stride_x, stride_x*sizeof(unsigned));
            result += stride_x;
            std::memcpy(result, y+j*stride_y, stride_y*sizeof(unsigned));
            result += stride_y;
        }
    }
}
void new_convol_filt_array(int K, css_whole& ncss, css_whole& ncss_new){
    //different order with python version
    int len_x = ncss.max_r;
    int len_y = ncss_new.max_r;

    unsigned old_lens[len_x+1];
    for(int i=0; i<len_x+1; i++){
        old_lens[i] = ncss.lens[i];
    }

    for(int i=0; i<len_x+1; i++){
        for(int j=0; j<len_y+1; j++){
            if((i+j)>K || j==0){
                continue;
            }
            if(i==0){
                std::memcpy(ncss.combs[j]+j*ncss.lens[j], ncss_new.combs[j], j*ncss_new.lens[j]*sizeof(unsigned));
                ncss.lens[j] += ncss_new.lens[j];

            }
            else{
                new_cross_join(ncss.combs[i+j]+(i+j)*ncss.lens[i+j], 
                                ncss.combs[i], old_lens[i], i, 
                                ncss_new.combs[j], ncss_new.lens[j], j);
                ncss.lens[i+j] += old_lens[i]*ncss_new.lens[j];
            }
        }
    }

}

