#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

#include "Configs.h"
#include "ModelPool.h"
#include "StatsMatrix.h"


struct AbcResult {
    IndelModelConfig winningConfig;
    Eigen::VectorXd meanParams; // 5 values: root, insRate, delRate, insLenParam, delLenParam
};

inline AbcResult abcInfer(const ModelPool& pool, const Eigen::VectorXd& empiricalStats, int topCutoff) {
    StatsMatrix sm{pool.pooledMatrix.rightCols(26)};
    Eigen::VectorXd dists = sm.mahalanobis(empiricalStats);

    // 1. vote using mixed top_cutoff across all models
    std::vector<int> voteIdx = sm.nsmallest(dists, topCutoff);

    std::vector<int> voteCounts(pool.configs.size(), 0);
    for (int idx : voteIdx) {
        voteCounts[pool.tags[idx]]++;
    }

    int winningTag = 0;
    for (size_t k = 1; k < pool.configs.size(); ++k) {
        if (voteCounts[k] > voteCounts[winningTag] ||
            (voteCounts[k] == voteCounts[winningTag] &&
             pool.configs[k].complexity() < pool.configs[winningTag].complexity())) {
            winningTag = static_cast<int>(k);
        }
    }

    // 2. re-rank within winning model's own rows only, take fresh top_cutoff
    std::vector<int> modelRows;
    for (size_t i = 0; i < pool.tags.size(); ++i) {
        if (pool.tags[i] == winningTag) {
            modelRows.push_back(static_cast<int>(i));
        }
    }

    int finalCutoff = std::min(topCutoff, static_cast<int>(modelRows.size()));
    std::partial_sort(modelRows.begin(), modelRows.begin() + finalCutoff, modelRows.end(),
        [&dists](int a, int b) { return dists(a) < dists(b); });
    modelRows.resize(finalCutoff);

    // 3. mean of param cols (0-4) over final rows
    Eigen::VectorXd meanParams = Eigen::VectorXd::Zero(NUM_PARAM_COLS);
    for (int idx : modelRows) {
        meanParams += pool.pooledMatrix.row(idx).head(NUM_PARAM_COLS).transpose();
    }
    meanParams /= double(modelRows.size());

    return AbcResult{pool.configs[winningTag], meanParams};
}