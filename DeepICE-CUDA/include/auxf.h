typedef std::vector<std::vector<unsigned>> cs;
struct CSVResult {
    std::vector<DATA_TYPE> rowMajorArray; // Holds the matrix in row-major order without the last column
    std::vector<LABEL_TYPE> lastColumn;   // Holds the last element of each row
};


CSVResult readCSVtoRowMajorAndLastColumn(const std::string& fileName, size_t& rows, size_t& cols);
void asgns_gen_rec(int8_t* asgns, int len, int K, int offset); 
unsigned combination(unsigned n, unsigned r); 
void generateDataset(int D, int N, double overlap, double balance_fac,
                     std::vector<std::vector<float>>& X, std::vector<int8_t>& t, unsigned int seed);
std::vector<float> linearize(const std::vector<std::vector<float>>& matrix); 
void print_2D_vector(const std::vector<std::vector<unsigned>>& vec); 

//std::vector<std::vector<std::vector<unsigned>>> kcombs_iter_array(int K, int n1, int n2);
cs cross_join_array(cs& cs1, cs& cs2);
std::vector<cs> convol_filt_array(cs (*f)(cs&, cs&), int k, std::vector<cs> x, std::vector<cs> y);
//void genCombTable(unsigned N, unsigned R);

css_whole reserve_space_css(int N, int D);
css_whole reserve_space_ncss(css_whole& css, int K);
void free_combs(css_whole& css);
void update_css(css_whole &css, int n);
void print_2D_array(const unsigned* arr, int rows, int cols);
css_whole new_kcombs_iter_array(int n1, int n2, int K);
void new_convol_filt_array(int k, css_whole& ncss, css_whole& ncss_new);