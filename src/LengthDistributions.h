#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

#include "Configs.h"



inline void buildLengthDistProbs(PriorConfig::LengthDist dist, double param, int truncation,
                                  std::vector<double>& out) {
    // zipf distribution
    if (dist == PriorConfig::LengthDist::Zipf) {
        double sum = 0.0;
        for (int i = 1; i <= truncation; ++i) {
            out[i - 1] = 1.0 / std::pow(i, param);
            sum += out[i - 1];
        }
        for (int i = 0; i < truncation; ++i) {
            out[i] /= sum;
        }
        return;
    }
    // Add other distributions here
    throw std::runtime_error("Length distribution not yet implemented");
}
 
inline std::vector<double> buildLengthDistProbs(PriorConfig::LengthDist dist, double param, int truncation) {
    std::vector<double> probs(truncation);
    buildLengthDistProbs(dist, param, truncation, probs);
    return probs;
}
 
