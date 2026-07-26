#pragma once
#include <vector>
#include <Eigen/Dense>

#include "Configs.h"
#include "SimulationBatch.h"


struct ModelPool {
    std::vector<IndelModelConfig> configs;
    const PriorConfig& priorConfig;
    int numSimsPerModel;
    unsigned seed;

    Eigen::MatrixXd pooledMatrix;
    std::vector<int> tags;

    void run(tree& tree_) {
        const int numCols = NUM_PARAM_COLS + 26;
        pooledMatrix.resize(configs.size() * numSimsPerModel, numCols);
        tags.clear();
        tags.reserve(configs.size() * numSimsPerModel);

        for (size_t k = 0; k < configs.size(); ++k) {
            SimulationBatch batch{configs[k], priorConfig, numSimsPerModel, seed + static_cast<unsigned>(k)};
            Eigen::MatrixXd blockResult = batch.run(tree_);

            pooledMatrix.middleRows(k * numSimsPerModel, numSimsPerModel) = blockResult;

            for (int r = 0; r < numSimsPerModel; ++r) {
                tags.push_back(static_cast<int>(k));
            }
        }
    }
};