#ifndef KNN_H
#define KNN_H

#include <string>
#include <vector>
#include "flower.h"

double distance(const Flower& a, const Flower& b);
double distance_manhattan(const Flower& a, const Flower& b);
double distance_cosine(const Flower& a, const Flower& b);

std::string predict(const Flower& query, const std::vector<Flower>& train, int K);
std::string predict_manhattan(const Flower& query, const std::vector<Flower>& train, int K);
std::string predict_cosine(const Flower& query, const std::vector<Flower>& train, int K);

void compute_minmax(const std::vector<Flower>& flowers, double min_vals[4], double max_vals[4]);
void apply_minmax(std::vector<Flower>& flowers, const double min_vals[4], const double max_vals[4]);

#endif