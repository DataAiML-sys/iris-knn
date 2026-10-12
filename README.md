# Iris KNN Classifier (C++)

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A from-scratch implementation of the K-Nearest Neighbors algorithm in C++17
for classifying Iris flowers, with 5-fold cross-validation, three distance
metrics, and min-max feature normalization.

## Motivation

Built to practice core machine learning concepts and modern C++ project
structure: separation of interface and implementation, reproducible experiments,
and correct train/test preprocessing (no data leakage).

## Dataset

The classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set):
150 samples, 4 features, 3 species (setosa, versicolor, virginica).

Features:
- sepal length
- sepal width
- petal length
- petal width

## Algorithm

For each query sample:

1. Compute the distance to every training sample.
2. Sort by distance.
3. Take the K nearest neighbors.
4. Return the species with the most votes.

Three distance metrics are compared:
- **Euclidean** - straight-line distance.
- **Manhattan** - sum of absolute differences.
- **Cosine** - 1 minus cosine similarity (angle between feature vectors).

## Preprocessing

Min-max normalization is applied **inside** each cross-validation fold:

1. Compute min/max on the training split only.
2. Apply the same min/max to both training and test splits.

This avoids data leakage - the test set never influences preprocessing.

## Cross-validation

5-fold cross-validation with a fixed random seed (42) for reproducibility.
Results are averaged across all 5 folds.

## Results

![KNN accuracy vs K](https://raw.githubusercontent.com/DataAiML-sys/iris-knn/main/plot.png)

| K  | Euclidean | Manhattan | Cosine |
|----|-----------|-----------|--------|
| 1  | 96.00%    | 94.00%    | 81.33% |
| 3  | 95.33%    | 95.33%    | 81.33% |
| 5  | 96.67%    | 94.67%    | 78.00% |
| 7  | 97.33%    | 96.00%    | 81.33% |
| 10 | 97.33%    | 95.33%    | 80.00% |
| 15 | 96.67%    | 96.00%    | 80.67% |

**Best:** Euclidean with K = 7 or K = 10 - 97.33%.

## Findings

- **Euclidean and Manhattan** benefit from normalization: both metrics
  become more stable across different K values.
- **Cosine distance drops sharply** after min-max normalization
  (from ~97% to ~80%). This is expected: cosine measures the *angle*
  between feature vectors and relies on the ratio between features.
  Min-max normalization compresses each feature independently,
  distorting those ratios. This is a known interaction between
  normalization and angle-based metrics.

## Build

```
clang++ -std=c++17 -Iinclude src/main.cpp src/knn.cpp -o iris
```

## Run

```
./iris
```

## Project structure

```
.
├── include/
│   ├── flower.h      # Flower struct
│   └── knn.h         # Function declarations
├── src/
│   ├── knn.cpp       # KNN implementation
│   └── main.cpp      # Entry point
├── iris.csv
└── README.md
```

## Tech

- C++17
- No external dependencies