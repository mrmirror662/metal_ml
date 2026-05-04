// Shared MNIST loader. Header-only, used by example/mlp_classification and
// example/cnn_classification. Returns images normalized to [0, 1] in row-major
// flat layout [N, 784]; reshape on the host or via cg::ReshapeNode as needed.

#pragma once

#include "cg.h"
#include "dataset_path.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mnist {

inline uint32_t read_be_u32(std::ifstream& f) {
    uint8_t b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
           (uint32_t(b[2]) << 8)  | uint32_t(b[3]);
}

struct Dataset {
    cg::Tensor       images;   // [N, 784]
    std::vector<int> labels;   // [N]
};

inline Dataset load(const std::string& images_path, const std::string& labels_path) {
    std::ifstream fi(images_path, std::ios::binary);
    std::ifstream fl(labels_path, std::ios::binary);
    if (!fi) throw std::runtime_error("cannot open " + images_path);
    if (!fl) throw std::runtime_error("cannot open " + labels_path);

    read_be_u32(fi); uint32_t n = read_be_u32(fi);
    uint32_t rows = read_be_u32(fi); uint32_t cols = read_be_u32(fi);
    read_be_u32(fl); read_be_u32(fl);

    int dim = (int)(rows * cols);
    std::vector<uint8_t> raw(n * dim);
    fi.read(reinterpret_cast<char*>(raw.data()), raw.size());
    std::vector<uint8_t> rl(n);
    fl.read(reinterpret_cast<char*>(rl.data()), n);

    cg::Tensor images({(int)n, dim});
    for (size_t i = 0; i < raw.size(); ++i) images.data[i] = raw[i] / 255.0f;
    std::vector<int> labels(n);
    for (uint32_t i = 0; i < n; ++i) labels[i] = rl[i];
    return {std::move(images), std::move(labels)};
}

inline Dataset load_train() {
    std::string base = MNIST_DIR;
    return load(base + "/train-images-idx3-ubyte", base + "/train-labels-idx1-ubyte");
}

inline Dataset load_test() {
    std::string base = MNIST_DIR;
    return load(base + "/t10k-images-idx3-ubyte",  base + "/t10k-labels-idx1-ubyte");
}

// labels [N] -> [N, n_classes] one-hot
inline cg::Tensor one_hot(const std::vector<int>& labels, int n_classes) {
    cg::Tensor t({(int)labels.size(), n_classes});
    for (size_t i = 0; i < labels.size(); ++i)
        t.data[i * n_classes + labels[i]] = 1.0f;
    return t;
}

} // namespace mnist
