#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iomanip>
#include "flower.h"
#include "knn.h"

int main() {
    std::ifstream file("iris.csv");
    if (!file.is_open()){
        std::cerr << "Error: cannot open iris.csv" << std::endl;
        return 1;
    }
    std::vector<Flower> flowers;
    std::string line;
    bool first_line = true;

    while(std::getline(file, line)){
        if(first_line){
            first_line = false;
            continue;
        }
    

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> parts;

        while (std::getline(ss, token, ',')){
            parts.push_back(token);
        }

        if (parts.size() == 5){
            Flower f;
            f.sepal_length = std::stod(parts[0]);
            f.sepal_width = std::stod(parts[1]);
            f.petal_length = std::stod(parts[2]);
            f.petal_width = std::stod(parts[3]);
            f.species = parts[4];
            flowers.push_back(f);
        }
    }

    file.close();

    std::cout << "Loaded samples: " << flowers.size() << std::endl;
    std::mt19937 g(42);
    std::shuffle(flowers.begin(), flowers.end(), g);

    const int folds = 5;
    const int n = (int)flowers.size();
    const int fold_size = n / folds;

    std::cout << "\n5-fold cross-validation (with min-max normalization):" << std::endl;
    for(int K : {1, 3, 5, 7, 10, 15}){
        std::vector<double> acc_euclidean;
        std::vector<double> acc_manhattan;
        std::vector<double> acc_cosine;

        for(int fold = 0; fold < folds; fold++){
            std::vector<Flower> train;
            std::vector<Flower> test;

            for(int i = 0; i < n; i ++){
                int current_fold = i / fold_size;
                if(current_fold == fold) {
                    test.push_back(flowers[i]);
                }
                else{
                    train.push_back(flowers[i]);
                }
            }

            double min_vals[4], max_vals[4];
            compute_minmax(train, min_vals, max_vals);
            apply_minmax(train, min_vals, max_vals);
            apply_minmax(test, min_vals, max_vals);

            int c_eucl = 0, c_manh = 0, c_cos = 0;
            for(int i = 0; i < (int)test.size(); i++){
                if(predict(test[i], train, K) == test[i].species) c_eucl++;
                if(predict_manhattan(test[i], train, K) == test[i].species) c_manh++;
                if(predict_cosine(test[i], train, K) == test[i].species) c_cos++;
            }

            acc_euclidean.push_back(100.0 * c_eucl / test.size());
            acc_manhattan.push_back(100.0 * c_manh / test.size());
            acc_cosine.push_back(100.0 * c_cos / test.size());
        }
        auto mean_of = [](const std::vector<double>& v){
            double s = 0;
            for(double x : v) s += x;
            return s / v.size();
        };

    std::cout << " K = " << std::setw(2) << K
        << " | euclidean: " << std::fixed << std::setprecision(2)
        << mean_of(acc_euclidean) << "%" << " | manhattan: " 
        << mean_of(acc_manhattan) << "%" << " | cosine: "
        << mean_of(acc_cosine) << "%" << std::endl;
    }
}