#include "knn.h"
#include <cmath>
#include <algorithm>
#include <utility>
#include <map>

double distance(const Flower& a, const Flower& b){
    double dl = a.sepal_length - b.sepal_length;
    double dw = a.sepal_width  - b.sepal_width;
    double pl = a.petal_length - b.petal_length;
    double pw = a.petal_width  - b.petal_width;

    return std::sqrt(dl * dl + dw * dw + pl * pl + pw * pw);
}
double distance_manhattan(const Flower& a, const Flower& b){
    return std::abs(a.sepal_length - b.sepal_length)
        + std::abs(a.sepal_width - b.sepal_width)
        + std::abs(a.petal_length - b.petal_length)
        + std::abs(a.petal_width - b.petal_width);
}
double distance_cosine(const Flower& a, const Flower& b){
    double dot = a.sepal_length * b.sepal_length
        + a.sepal_width  * b.sepal_width
        + a.petal_length * b.petal_length
        + a.petal_width  * b.petal_width;
    
    double len_a = std::sqrt(a.sepal_length * a.sepal_length
        + a.sepal_width * a.sepal_width
        + a.petal_length * a.petal_length
        + a.petal_width * a.petal_width);

    double len_b = std::sqrt(b.sepal_length * b.sepal_length
        + b.sepal_width * b.sepal_width
        + b.petal_length * b.petal_length
        + b.petal_width * b.petal_width);
    
    return 1.0 - dot / (len_a * len_b);
}

std::string predict(const Flower& query, const std::vector<Flower>& train, int K){
    std::vector<std::pair<double, int>> dists;
    for (int i = 0; i < (int)train.size(); i ++){
        double d = distance(query, train[i]);
        dists.push_back({d, i});
    }
   
    std::sort(dists.begin(), dists.end());

    std::map<std::string, int> votes;
    for (int i = 0; i < K && i < (int)dists.size(); i++) {
        int idx = dists[i].second;
        votes[train[idx].species]++;
    }

    std::string best;
    int best_count = -1;
    for (auto& [species, count] : votes) {
        if (count > best_count) {
            best_count = count;
            best = species;
        }
    }

    return best;
}
std::string predict_manhattan(const Flower& query, const std::vector<Flower>& train, int K) {
    std::vector<std::pair<double, int>> dists;
    for (int i = 0; i < (int)train.size(); i++) {
        double d = distance_manhattan(query, train[i]);
        dists.push_back({d, i});
    }

    std::sort(dists.begin(), dists.end());

    std::map<std::string, int> votes;
    for (int i = 0; i < K && i < (int)dists.size(); i++) {
        int idx = dists[i].second;
        votes[train[idx].species]++;
    }

    std::string best;
    int best_count = -1;
    for (auto& [species, count] : votes) {
        if (count > best_count) {
            best_count = count;
            best = species;
        }
    }

    return best;
}
std::string predict_cosine(const Flower& query, const std::vector<Flower>& train, int K) {
    std::vector<std::pair<double, int>> dists;
    for (int i = 0; i < (int)train.size(); i++) {
        double d = distance_cosine(query, train[i]);
        dists.push_back({d, i});
    }

    std::sort(dists.begin(), dists.end());

    std::map<std::string, int> votes;
    for (int i = 0; i < K && i < (int)dists.size(); i++) {
        int idx = dists[i].second;
        votes[train[idx].species]++;
    }

    std::string best;
    int best_count = -1;
    for (auto& [species, count] : votes) {
        if (count > best_count) {
            best_count = count;
            best = species;
        }
    }
    return best;
}
void compute_minmax(const std::vector<Flower>& flowers, double min_vals[4], double max_vals[4]){
    min_vals[0] = max_vals[0] = flowers[0].sepal_length;
    min_vals[1] = max_vals[1] = flowers[0].sepal_width;
    min_vals[2] = max_vals[2] = flowers[0].petal_length;
    min_vals[3] = max_vals[3] = flowers[0].petal_width;

    for(const auto& f :flowers){
        if(f.sepal_length < min_vals[0]) min_vals[0] = f.sepal_length;
        if(f.sepal_length > max_vals[0]) max_vals[0] = f.sepal_length;
        if(f.sepal_width < min_vals[1]) min_vals[1] = f.sepal_width;
        if(f.sepal_width > max_vals[1]) max_vals[1] = f.sepal_width;
        if(f.petal_length < min_vals[2]) min_vals[2] = f.petal_length;
        if(f.petal_length > max_vals[2]) max_vals[2] = f.petal_length;
        if(f.petal_width < min_vals[3]) min_vals[3] = f.petal_width;
        if(f.petal_width > max_vals[3]) max_vals[3] = f.petal_width;
    }
}
void apply_minmax(std::vector<Flower>& flowers, const double min_vals[4], const double max_vals[4]) {
    for (auto& f : flowers) {
        f.sepal_length = (f.sepal_length - min_vals[0]) / (max_vals[0] - min_vals[0]);
        f.sepal_width  = (f.sepal_width  - min_vals[1]) / (max_vals[1] - min_vals[1]);
        f.petal_length = (f.petal_length - min_vals[2]) / (max_vals[2] - min_vals[2]);
        f.petal_width  = (f.petal_width  - min_vals[3]) / (max_vals[3] - min_vals[3]);
    }
}