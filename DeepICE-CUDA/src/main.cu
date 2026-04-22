#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <getopt.h>
#include "types.h"
#include "auxf.h"
#include "gpu_utilities.h"
#include "deep_ice_chez.h"

std::vector<std::vector<double>> read_matrix(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::string line;

    if (!file) {
        std::cerr << "Error: Cannot open file!" << std::endl;
        return matrix;
    }

    while (std::getline(file, line)) {  // Read each line
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {  // Read numbers in line
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }

    return matrix;
}

// ----------------- Main Function -----------------
int main(int argc, char* argv[]) {
    int K = NUM_RELUS;

    std::string inputFile = "../dataset/voicepath_data.csv";
    bool shuffle_flag = false;
    std::string shuffle_setting_file;

    int L = 30, R=1, Nremain=60;
    int shuffle_block_size = 10;
    int best_candidates_size = 500;

    struct option long_options[] = {
        {"input", required_argument, nullptr, 'i'}, // --range requires an argument
        {nullptr, 0, nullptr, 0} // End of array marker
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                inputFile = optarg;
                break;
        }
    }


    size_t rows = 0, cols = 0;
    CSVResult data = readCSVtoRowMajorAndLastColumn(inputFile, rows, cols);

    
    int N = rows, D = cols;
    std::cout<<"N: "<< N<< " D: "<<D <<" K: "<<K<<std::endl;

    // auto res = Deep_ICE(N, D, K, data.rowMajorArray.data(), data.lastColumn.data());
    cnfg res = Chez_ICE(N, D, K, data.rowMajorArray.data(), data.lastColumn.data());

    std::cout << "The final best ncss is: ";
    for (auto &ele : res.comb)
    {
        std::cout << ele << ",";
    }

    std::cout << " it has 0-1 loss: " << res.loss << std::endl;

    return 0;
}
